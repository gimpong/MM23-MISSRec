import torch
try:
  import cupy as cp
except ModuleNotFoundError:
  raise ModuleNotFoundError("cupy is not installed, please visit https://pypi.org/project/cupy/")

from . import clustering
