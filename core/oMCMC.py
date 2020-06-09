import numpy as np
import torch
from core.kernels import LeapFrog


class Normal:
    def __init__(self, sigma, dim, device):
        self.sigma = sigma
        self.dim = dim
        self.device = device

    def log_prob(self, x):
        assert (x.shape[1] == self.dim)
        mu = 0.0
        return -0.5 * torch.sum((x - mu) ** 2, dim=1) / self.sigma ** 2 - 0.5 * self.dim * np.log(
            2 * np.pi * self.sigma ** 2)

    def sample(self, n):
        return self.sigma * torch.empty([n, self.dim]).normal_()


def HMC(target, dim, step_size, n_steps, device):
    momentum = Normal(1.0, dim, device)
    LF = LeapFrog(target, momentum, step_size, dim, device)
    
    def kernel(x_0, v_0, d):
        assert (x_0.shape[1] == dim) and (v_0.shape[1] == dim)
        # iterating over orbit
        x, v, log_probs = LF.iterate_n(x_0, v_0, n_steps)
        
        # switching between orbits
        log_p = (log_probs[:,n_steps] - log_probs[:,0]).flatten()
        log_u = torch.log(torch.zeros_like(log_p).uniform_().to(device))
        accepted_mask = (log_p > log_u).float()
        accepted_mask = accepted_mask[:,np.newaxis]
        next_x = x[:,:,-1]*accepted_mask + x[:,:,0]*(1-accepted_mask)
        next_state = (next_x, momentum.sample(x.shape[0]), d)
        
        weights = torch.ones([x.shape[0],1]).to(device)
        samples = next_x
        return samples[:,:,np.newaxis], weights, next_state, accepted_mask
    
    def init_generator(batch_size):
        return torch.zeros([batch_size, dim]), momentum.sample(batch_size)
    
    return kernel, init_generator


def batch_categorical(probs, device):
    batch_size = probs.shape[0]
    u = torch.empty([batch_size,1]).uniform_().to(device)
    cums = torch.cumsum(probs, dim=1) - u
    cums[cums > 0.0] = 0.0
    ids = torch.sum(-torch.sign(cums), dim=1).long()
    return ids


def oHMC(target, dim, step_size, n_steps, device):
    momentum = Normal(1.0, dim, device)
    LF = LeapFrog(target, momentum, step_size, dim, device)
    
    def kernel(x_0,v_0,d):
        assert (x_0.shape[1] == dim) and (v_0.shape[1] == dim)
        # iterating over orbit
        x, v, log_probs = LF.iterate_n_directed(x_0, v_0, n_steps, d)
        
        # accepting the orbit
        samples = x
        weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True))
        
        # switching between orbits
        next_d = batch_categorical(weights, device)
        ids = next_d.view(-1,1).repeat([1,dim]).view([x_0.shape[0],dim,1])
        next_x = x.gather(2, ids).squeeze(2)
        next_state = (next_x, momentum.sample(x.shape[0]), (next_d+1) % (n_steps+1))
        
        log_p = (log_probs[:,n_steps] - log_probs[:,0]).flatten()
        log_u = torch.log(torch.empty_like(log_p).uniform_().to(device))
        accepted_mask = (log_p > log_u).float()
        return samples, weights, next_state, accepted_mask
    
    def init_generator(batch_size):
        return torch.zeros([batch_size, dim]), momentum.sample(batch_size)
    
    return kernel, init_generator

def oHMC_approx(target, dim, step_size, n_steps, device):
    momentum = Normal(1.0, dim, device)
    LF = LeapFrog(target, momentum, step_size, dim, device)
    
    def kernel(x_0,v_0,d):
        assert (x_0.shape[1] == dim) and (v_0.shape[1] == dim)
        # iterating over orbit
        x, v, log_probs = LF.iterate_n(x_0, v_0, n_steps)
        
        # accepting the orbit
        samples = x
        weights = torch.exp(log_probs)/torch.sum(torch.exp(log_probs), dim=1, keepdim=True)
        
        # switching between orbits
        ids = batch_categorical(weights, device)
        ids = ids.view(-1,1).repeat([1,dim]).view([x_0.shape[0],dim,1])
        next_x = x.gather(2, ids).squeeze(2)
        next_state = (next_x, momentum.sample(x.shape[0]), d)
        
        log_p = (log_probs[:,n_steps] - log_probs[:,0]).flatten()
        log_u = torch.log(torch.empty_like(log_p).uniform_().to(device))
        accepted_mask = (log_p > log_u).float()
        accepted_mask = accepted_mask[:,np.newaxis]
#         next_x = x[:,:,-1]*accepted_mask + x[:,:,0]*(1-accepted_mask)
#         next_state = (next_x, momentum.sample(x.shape[0]), d)
        return samples, weights, next_state, accepted_mask
    
    def init_generator(batch_size):
        return torch.zeros([batch_size, dim]), momentum.sample(batch_size)
    
    return kernel, init_generator
