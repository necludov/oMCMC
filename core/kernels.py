import numpy as np
import torch


class LeapFrog():
    def __init__(self, target, momentum, step_size, dim, device):
        self.target = target
        self.momentum = momentum
        self.eps = step_size
        self.dim = dim
        self.device = device
        
    def iterate_coordinate_verlet(self, x_0, v_0):
        assert (x_0.shape[1] == v_0.shape[1] == self.dim)
        x_1 = x_0 - 0.5*self.eps*(-v_0)
        v_1 = v_0 + self.eps*self.evaluate_grad(x_1)
        x_2 = x_1 - 0.5*self.eps*(-v_1)
        return x_2, v_1
    
    def iterate_coordinate_verlet_inv(self, x_0, v_0):
        assert (x_0.shape[1] == v_0.shape[1] == self.dim)
        x_1 = x_0 + 0.5*self.eps*(-v_0)
        v_1 = v_0 - self.eps*self.evaluate_grad(x_1)
        x_2 = x_1 + 0.5*self.eps*(-v_1)
        return x_2, v_1
    
    def iterate(self, x_0, v_0):
        assert (x_0.shape[1] == v_0.shape[1] == self.dim)
        v_1 = v_0 + 0.5*self.eps*self.evaluate_grad(x_0)
        x_1 = x_0 - self.eps*(-v_1)
        v_2 = v_1 + 0.5*self.eps*self.evaluate_grad(x_1)
        return x_1, v_2
    
    def iterate_inv(self, x_0, v_0):
        assert (x_0.shape[1] == v_0.shape[1] == self.dim)
        v_1 = v_0 - 0.5*self.eps*self.evaluate_grad(x_0)
        x_1 = x_0 + self.eps*(-v_1)
        v_2 = v_1 - 0.5*self.eps*self.evaluate_grad(x_1)
        return x_1, v_2
    
    def iterate_n_directed(self, x_0, v_0, n_steps, d):
        assert (x_0.shape[1] == v_0.shape[1] == self.dim)
        batch_size = x_0.shape[0]
        x = torch.zeros([batch_size, self.dim, 2*n_steps + 1], requires_grad=False).to(self.device)
        v = torch.zeros([batch_size, self.dim, 2*n_steps + 1], requires_grad=False).to(self.device)
        log_probs = torch.zeros([batch_size, 2*n_steps + 1], requires_grad=False).to(self.device)
        i_0 = n_steps
        x[:,:,i_0], v[:,:,i_0] = x_0, v_0
        log_probs[:,i_0] = self.target.log_prob(x[:,:,i_0]) + self.momentum.log_prob(v[:,:,i_0])
        for i in range(n_steps):
            x[:,:,i_0+i+1], v[:,:,i_0+i+1] = self.iterate(x[:,:,i_0+i],v[:,:,i_0+i])
            log_probs[:,i_0+i+1] = self.target.log_prob(x[:,:,i_0+i+1]) + self.momentum.log_prob(v[:,:,i_0+i+1])
        for i in range(n_steps):
            x[:,:,i_0-i-1], v[:,:,i_0-i-1] = self.iterate_inv(x[:,:,i_0-i],v[:,:,i_0-i])
            log_probs[:,i_0-i-1] = self.target.log_prob(x[:,:,i_0-i-1]) + self.momentum.log_prob(v[:,:,i_0-i-1])
        d = i_0-d
        d = d.view(-1,1) + torch.arange(n_steps+1).view(1,-1).to(self.device)
        d_ = d.clone()
        d = d.view(batch_size,1,n_steps+1).repeat([1,self.dim,1]).view([batch_size,self.dim,n_steps+1])
        x_shifted = x.gather(2,d.long())
        v_shifted = v.gather(2,d.long())
        log_probs_shifted = log_probs.gather(1,d_.long())
#         x_shifted = torch.cat([x[chain_id:chain_id+1,:,i_0-ch_d:i_0-ch_d+n_steps+1] for chain_id, ch_d in enumerate(d)], dim=0)
#         v_shifted = torch.cat([v[chain_id:chain_id+1,:,i_0-ch_d:i_0-ch_d+n_steps+1] for chain_id, ch_d in enumerate(d)], dim=0)
#         log_probs_shifted = torch.cat([log_probs[chain_id:chain_id+1,i_0-ch_d:i_0-ch_d+n_steps+1] for chain_id, ch_d in enumerate(d)], dim=0)
#         x_shifted = torch.zeros([x_0.shape[0], x_0.shape[1], n_steps + 1]).to(self.device)
#         v_shifted = torch.zeros([v_0.shape[0], v_0.shape[1], n_steps + 1]).to(self.device)
#         log_probs_shifted = torch.zeros([x_0.shape[0], n_steps + 1]).to(self.device)
#         for chain_id in range(x.shape[0]):
#             x_shifted[chain_id,:,:] = x[chain_id,:,i_0-d[chain_id]:i_0-d[chain_id]+n_steps+1]
#             v_shifted[chain_id,:,:] = v[chain_id,:,i_0-d[chain_id]:i_0-d[chain_id]+n_steps+1]
#             log_probs_shifted[chain_id,:] = log_probs[chain_id,i_0-d[chain_id]:i_0-d[chain_id]+n_steps+1]
        return x_shifted, v_shifted, log_probs_shifted
    
    
    def iterate_n(self, x_0, v_0, n_steps):
        assert (x_0.shape[1] == v_0.shape[1] == self.dim)
        batch_size = x_0.shape[0]
        x = torch.zeros([batch_size, self.dim, n_steps + 1], requires_grad=False).to(self.device)
        v = torch.zeros([batch_size, self.dim, n_steps + 1], requires_grad=False).to(self.device)
        log_probs = torch.zeros([batch_size, n_steps + 1], requires_grad=False).to(self.device)
        x[:,:,0], v[:,:,0] = x_0, v_0
        log_probs[:,0] = self.target.log_prob(x[:,:,0]) + self.momentum.log_prob(v[:,:,0])
        for i in range(n_steps):
            x[:,:,i+1], v[:,:,i+1] = self.iterate(x[:,:,i],v[:,:,i])
            log_probs[:,i+1] = self.target.log_prob(x[:,:,i+1]) + self.momentum.log_prob(v[:,:,i+1])
        return x, v, log_probs
    
    def evaluate_grad(self, x):
        y = x.detach()
        y.requires_grad = True
        grad = torch.autograd.grad(self.target.log_prob(y), y, torch.ones([y.shape[0]]).to(self.device))[0]
        assert (grad.shape[0] == x.shape[0]) and (grad.shape[1] == self.dim)
        return grad.detach()
