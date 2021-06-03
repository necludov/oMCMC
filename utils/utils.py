import torch
import numpy as np

def get_random_batch(dataset, k=1):
    n, H, W = dataset.data.shape
    labels = dataset.targets.unique()
    im_batch = torch.zeros([len(labels)*k, 1, H, W])
    label_batch = torch.zeros([len(labels)*k], dtype=torch.long)
    for label_id in range(len(labels)):
        obj_id = torch.multinomial((dataset.targets == labels[label_id]).float(), k)
        im_batch[label_id*k:(label_id+1)*k,0,:,:] = dataset.data[obj_id]
        label_batch[label_id*k:(label_id+1)*k] = labels[label_id]
    return im_batch, label_batch

def batch_means_ess(x):
    """ Estimate the effective sample size as the ratio of the variance
    of the batch means to the variance of the chain. As explained here:
    https://arxiv.org/pdf/1011.0175.pdf. We expect the chain in the format
    Time-Steps, Num-Chains, Dimension (T, M, D) """

    x = np.transpose(x, [2, 0, 1]) # I expect Num-Chains, Dimension, Time-Steps
    T, M, D = x.shape
    num_batches = int(np.floor(T ** (1 / 3)))
    batch_size = int(np.floor(num_batches ** 2))
    batch_means = []
    for i in range(num_batches):
        batch = x[batch_size * i:batch_size * i + batch_size]
        batch_means.append(np.mean(batch, axis=0))
    batch_variance = np.var(np.array(batch_means), axis=0)
    chain_variance = np.var(x, axis=0)

    act = batch_size * batch_variance / (chain_variance + 1e-20)

    return 1 / act

def batch_means_ess_weighted(x,w):
    """ Estimate the effective sample size as the ratio of the variance
    of the batch means to the variance of the chain. As explained here:
    https://arxiv.org/pdf/1011.0175.pdf. We expect the chain in the format
    Time-Steps, Num-Chains, Dimension (T, M, D) and the weights in the format
    Time-Steps, Num-Chains, Dimension (T, M)"""
    
    x = np.transpose(x, [2, 0, 1]) # I expect Num-Chains, Dimension, Time-Steps
    T, M, D = x.shape
    weights = w[:,:,np.newaxis].copy()
    weights = weights/np.sum(weights, axis=0, keepdims=True)
    num_batches = int(np.floor(T ** (1 / 3)))
    batch_size = int(np.floor(num_batches ** 2))
    batch_means = []
    for i in range(num_batches):
        batch = x[batch_size * i:batch_size * i + batch_size]
        batch_weights = weights[batch_size * i:batch_size * i + batch_size].copy()
        batch_weights /= np.sum(batch_weights, axis=0, keepdims=True)
        batch_means.append(np.sum(batch*batch_weights, axis=0))
    batch_variance = np.var(np.array(batch_means), axis=0)
    chain_mean = np.sum(x*weights, axis=0, keepdims=True)
    chain_variance = np.sum(weights*(x-chain_mean)**2, axis=0)
    act = batch_size * batch_variance / (chain_variance+1e-20)
    return 1/act
