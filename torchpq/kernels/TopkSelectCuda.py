import torch
import cupy as cp
import numpy as np
import math

from .CustomKernel import CustomKernel
from ..util import get_absolute_path

class TopkSelectCuda(CustomKernel):
  """
    tpb: threads per block, needs to be a power of 2 between 32 and 1024
    queue_capacity: capacity of thread queue
    buffer_size: number of elements each threads needs to prefetch
  """
  def __init__(self, tpb=256, queue_capacity=4, buffer_size=4):
    super().__init__()
    assert tpb >= 32
    assert self.next_power_of_2(tpb) == tpb
    assert queue_capacity >= 1
    assert buffer_size >= 1
    self.tpb = tpb
    self.queue_capacity = queue_capacity
    self.buffer_size = buffer_size

    with open(get_absolute_path("kernels", "cuda", "topk_select.cu"),'r') as f: ###
      self.kernel = f.read()
    
    self.kernel = (
      self.kernel
      .replace("_TPB_", str(tpb))
      .replace("_QCAP_", str(queue_capacity))
      .replace("_TN_", str(buffer_size))
    )

    self._fn = cp.RawKernel(
      code=self.kernel,
      name="topk_select",
      backend='nvrtc',
      options=(
        '--use_fast_math',
        "-lineinfo",
        # '--maxrregcount=128',
        #'-Xptxas',
        #'-dlcm=cg',
      )
    )

  @staticmethod
  def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
  
  def __call__(self, x, k=128, dim=1):
    """
      x: shape = [m, n], dtype: float32
      k: 1 to 1024
      dim: 1
    """
    assert len(x.shape) == 2
    assert x.dtype in [torch.float32]
    assert x.device.type == "cuda"
    assert 1 <= k <= self.tpb
    assert dim == 1
    assert x.is_contiguous()
    k_pow_of_2 = self.next_power_of_2(k)
    device = x.device

    m, n = x.shape
    threads_per_block = (self.tpb, )
    blocks_per_grid = (m, )
    values = torch.empty(m, k_pow_of_2, device=device, dtype=x.dtype)
    values.fill_(float("-inf"))
    indices = torch.empty(m, k_pow_of_2, device=device, dtype=torch.long)

    self._fn(
      grid = blocks_per_grid,
      block = threads_per_block,
      args = [
        x.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        m, n, k_pow_of_2
      ],
      stream=self.stream
    )
    return values[:, :k], indices[:, :k]