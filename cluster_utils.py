import numpy as np
import torch
from copy import deepcopy
from torchpq.clustering import MinibatchKMeans
from tqdm import tqdm
import logging

def cluster_dpc_knn(all_tokens, cluster_num, local_size=5, token_weight=None, device=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        cluster_idx (Tensor[N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        all_tokens (dict): dict for token information
        cluster_num (int): cluster number
        local_size (int): number of the nearest neighbor used for local density.
        token_weight (Tensor[N]): weight for each token.
    """
    assert all_tokens.ndim == 2
    N, D = all_tokens.shape
    with torch.no_grad():
        if device:
            saved_device = all_tokens.device
            all_tokens = all_tokens.to(device)

        dist_matrix = torch.cdist(all_tokens, all_tokens) / (D ** 0.5) # [NxN]

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=local_size, dim=-1, largest=False) # [Nxk]

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp() # [N]
        # add a little noise to ensure no tokens have the same density.
        # density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

        # get distance indicator
        mask = density.unsqueeze(0) > density.unsqueeze(-1) # [NxN]
        dist_max = dist_matrix.max()
        dist, index_parent = (dist_matrix * mask + dist_max * (~mask)).min(dim=-1) # [N]

        # select clustering center according to score
        score = dist * density # [N]
        _, index_down = torch.topk(score, k=cluster_num, dim=-1) # index_down: [Nc]

        # assign tokens to the nearest center
        dist_matrix = dist_matrix[index_down] # [NcxN]
        cluster_idx = dist_matrix.argmin(dim=0) # [N]

        # make sure cluster center merge to itself
        idx_tmp = torch.arange(cluster_num, device=all_tokens.device) # [Nc]
        cluster_idx[index_down] = idx_tmp # [N]
        
        if token_weight is None:
            token_weight = all_tokens.new_ones(N)

        # compute token-wise weights to cluster centroids
        all_weight = token_weight.new_zeros(cluster_num) # [Nc]
        all_weight.index_add_(dim=0, index=cluster_idx, source=token_weight) # [Nc]
        all_weight = all_weight + 1e-4 # [Nc]
        norm_weight = token_weight / all_weight[cluster_idx] # [N] / [Nc[N]] => [N]

        # average token features
        centroids = all_tokens.new_zeros(cluster_num, D) # [NcxD]
        weighted_all_tokens = all_tokens * norm_weight.unsqueeze(-1) # [NxD] * [Nx1] => [NxD]
        centroids.index_add_(dim=0, index=cluster_idx, source=weighted_all_tokens) # [NcxD]
        
        # NOTE: PLEASE REMEMBER TO ADD 1 AS OFFSET FOR EMBEDDING LAYER
        if device:
            ret_cluster_idx = cluster_idx.to(saved_device)
            ret_centroids = centroids.to(saved_device)
            return ret_cluster_idx, ret_centroids
        else:
            return cluster_idx, centroids

def cluster_kmeans(all_tokens, cluster_num, batch_size=2048, niter=5, tol=1e-5):
    # compute the batchsize of last batch
    if batch_size <= cluster_num:
        new_batch_size = int(cluster_num * 1.1)
        print(f"adjust batchsize from {batch_size} (given) to {new_batch_size} because the cluster_num = {cluster_num}")
        batch_size = new_batch_size
    
    kmeans = MinibatchKMeans(n_clusters=cluster_num)
    for i in tqdm(range(niter), 'clustering iter'):
        for ptr in range(0, len(all_tokens), batch_size):
            remaining_size = len(all_tokens) - ptr - batch_size
            # if the last batch is smaller than cluster_num, then merge it to the second last batch
            if remaining_size <= cluster_num:
                offset = batch_size + remaining_size
            else:
                offset = batch_size
            batch_tokens = all_tokens[ptr: ptr+offset].T
            kmeans.fit_minibatch(batch_tokens)
            if remaining_size <= cluster_num:
                break
        if kmeans.error < tol:
            break

    batch_cluster_idx = []
    for ptr in range(0, len(all_tokens), batch_size):
        batch_cluster_idx.append(kmeans.predict(all_tokens[ptr: ptr+batch_size].T))
    cluster_idx = torch.cat(batch_cluster_idx, dim=0)
    centroids = deepcopy(kmeans.centroids.T)
    del kmeans
    del batch_cluster_idx
    return cluster_idx, centroids

if __name__ == '__main__':
    N = 446975*2
    D = 300
    cluster_num = int(N * 0.1)
    batch_size = 2048
    niter = 3
    all_tokens = torch.rand(N, D, device='cuda:0')
    cluster_idx, centroids = cluster_kmeans(all_tokens, cluster_num, batch_size, niter)
    print(cluster_idx.shape, centroids.shape)
    input("wait")