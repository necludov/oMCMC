import numpy as np
import torch
from core import integrators
from core import distributions

class HMC:
    def __init__(self, target, trajectory_length=None, step_size=None, alpha=1.0, jitter=True):
        self.device = target.device
        self.momentum = distributions.Normal(torch.zeros(target.dim, device=self.device), 1.0, self.device)
        self.target = target
        self.eps = step_size
        self.T_max = trajectory_length
        self.alpha = alpha
        self.jitter = jitter
        
    def set_n_steps(self):
        T = self.T_max
        if self.jitter:
            T = torch.ones([1], device=self.device).uniform_()*self.T_max
        self.n_steps = torch.ceil(T/self.eps).long()
        return self.n_steps, T
        
    @torch.no_grad()
    def log_prob(self, x, v):
        return self.target.log_prob(x) + self.momentum.log_prob(v)
        
    def find_reasonable_eps(self, x_0):
        batch_size = x_0.shape[0]
        eps = torch.ones([1], device=self.device, dtype=torch.float)
        v_0 = self.momentum.sample(batch_size)
        log_p_0 = self.log_prob(x_0, v_0)
        
        x_1, v_1, log_p_1, _ = integrators.leap_frog(x_0, v_0, eps, self.target, self.momentum)
        log_p = log_p_1 - log_p_0
        acceptance_prob = torch.exp(torch.minimum(log_p, torch.zeros_like(log_p)))
        acceptance_prob = (1./acceptance_prob).mean()**(-1)
        power = -1*(acceptance_prob <= 0.5)+1*(acceptance_prob > 0.5)
        while acceptance_prob**power > 0.5**power:
            eps = eps*2.0**power
            x_1, v_1, log_p_1, _ = integrators.leap_frog(x_0, v_0, eps, self.target, self.momentum)
            log_p = log_p_1 - log_p_0
            acceptance_prob = torch.exp(torch.minimum(log_p, torch.zeros_like(log_p)))
            acceptance_prob = (1./acceptance_prob).mean()**(-1)
        return eps
        
    def iterate(self, x_0, v_0):
        batch_size = x_0.shape[0]
        
        log_p_0 = self.log_prob(x_0, v_0)
        x, v, log_p_last = x_0.clone(), v_0.clone(), log_p_0.clone()
        for step_i in range(self.n_steps):
            x, v, log_p_last, _ = integrators.leap_frog(x, v, self.eps, self.target, self.momentum)
        log_p = (log_p_last - log_p_0).flatten()
        log_u = torch.log(torch.zeros_like(log_p).uniform_())
        accepted_mask = (log_p > log_u).float()
        accepted_mask = accepted_mask.reshape([batch_size,1])
        next_x = x*accepted_mask + x_0*(1-accepted_mask)
        next_v = v*accepted_mask + v_0*(1-accepted_mask)
        next_state = (next_x, next_v)
        
        samples = next_x 
        log_accept_p = torch.minimum(log_p, torch.zeros_like(log_p))
        return samples, next_state, log_accept_p
    
    def adapt(self, x_0, n):
        batch_size = x_0.shape[0]
        self.eps = self.find_reasonable_eps(x_0)
        mu = torch.log(10*self.eps)
        gamma = 0.05
        time_0 = 1.0
        kappa = 0.75
        running_eps = torch.zeros_like(self.eps)
        running_ar = torch.zeros_like(self.eps)
        
        self.T_max = self.eps
        log_T = torch.nn.Parameter(torch.zeros(1, dtype=torch.float, device=self.device))
#         log_T = torch.nn.Parameter(torch.log(self.T_max))
        running_T = torch.zeros_like(log_T)
        optimizer = torch.optim.Adam([log_T], lr=0.025, betas=(0.0,0.95))
        grad_evals = 0
        
        x_next = x_0
        for i in range(1,n+1):
            n_steps, t = self.set_n_steps()
            
            v = self.momentum.sample(batch_size)
            x = x_next
            samples, next_state, log_accept_p = self.iterate(x, v)
            x_next, v_next = next_state
            
            # dual averaging of step
            acceptance_prob = torch.exp(log_accept_p).mean()
            running_ar = (1.-1./(i+time_0))*running_ar + 1./(i+time_0)*(0.65-acceptance_prob)
            log_eps = mu - np.sqrt(i)/gamma*running_ar
#             running_eps = i**(-kappa)*torch.exp(log_eps)+(1.-i**(-kappa))*running_eps
            running_eps = 0.9*running_eps+0.1*torch.exp(log_eps)
            self.eps = torch.exp(log_eps)
            
            # setting trajectory length by optimizing ChEES 
            acceptance_prob = torch.exp(log_accept_p).mean()
            x_mean, x_next_mean = x.mean(0, keepdim=True), x_next.mean(0, keepdim=True)
            grad_estimate = ((x_next - x_next_mean)*v_next).sum(1, keepdim=True)
            grad_estimate *= t*((x_next - x_next_mean).norm(2,dim=1,keepdim=True)-(x - x_mean).norm(2,dim=1,keepdim=True))
            grad_estimate = (torch.exp(log_accept_p)*grad_estimate).sum()/(torch.exp(log_accept_p).sum()+1e-5)
            optimizer.zero_grad()
            log_T.grad = -grad_estimate*torch.exp(log_T).detach()
            optimizer.step()
            self.T_max = torch.exp(log_T).detach()
            running_T = 0.9*running_T + 0.1*torch.exp(log_T).detach()
            
            grad_evals += n_steps
        self.eps = running_eps
        self.T_max = running_T
        return x, grad_evals
    
    def sample_n(self, x_0, n_iterations, budget=None):
        assert (x_0.shape[1] == self.target.dim)
        batch_size = x_0.shape[0]
        all_samples = torch.zeros([batch_size, self.target.dim, n_iterations], device=self.device)
        acceptance_rates = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        trajectories_l = torch.zeros(n_iterations, dtype=torch.long, device=self.device)
        grad_evals = 0
        iterations_done = 0
        
        v_0 = self.momentum.sample(batch_size)
        x, v = x_0, v_0
        for i in range(n_iterations):
            n_steps, t = self.set_n_steps()
            trajectories_l[i] = n_steps+1
            v = np.sqrt(1-self.alpha)*v+np.sqrt(self.alpha)*self.momentum.sample(batch_size)
            samples, next_state, log_accept_p = self.iterate(x, v)
            all_samples[:,:,i] = samples
            x, v = next_state
            acceptance_rates += torch.exp(log_accept_p)
            grad_evals += n_steps
            iterations_done += 1
            if (budget is not None) and (budget < grad_evals):
                break
            
        acceptance_rates /= iterations_done
        all_weights = torch.ones([batch_size, 1, n_iterations], device=self.device)
        return all_samples, all_weights, acceptance_rates, grad_evals, trajectories_l
    
    def subsample(self, samples, weights, n):
        ids = torch.multinomial(weights.squeeze(1), n)
        ids, _ = torch.sort(ids, dim=1)
        ids = ids.unsqueeze(1).repeat([1,self.target.dim,1])
        return samples.gather(2, ids)
    

class HMCRec:
    def __init__(self, target, trajectory_length, step_size, alpha=1.0, jitter=True):
        self.device = target.device
        self.momentum = distributions.Normal(torch.zeros(target.dim, device=self.device), 1.0, self.device)
        self.target = target
        self.eps = step_size
        self.T_max = trajectory_length
        self.jitter = jitter
        self.alpha = alpha
        
    def set_n_steps(self):
        T = self.T_max
        if self.jitter:
            T = torch.ones([1], device=self.device).uniform_()*T
        self.n_steps = torch.ceil(T/self.eps).long()
        return self.n_steps, T
    
    def max_n_steps(self):
        n_steps = torch.ceil(self.T_max/self.eps).long()
        return n_steps
        
    @torch.no_grad()
    def log_prob(self, x, v):
        return self.target.log_prob(x) + self.momentum.log_prob(v)
        
    def iterate(self, x_0, v_0):
        batch_size = x_0.shape[0]
        n = self.n_steps
        x = torch.zeros([batch_size, self.target.dim, n + 1], device=self.device)
        v = torch.zeros([batch_size, self.target.dim, n + 1], device=self.device)
        log_p = torch.zeros([batch_size, 1, n + 1], device=self.device)
        
        log_p[:,0,0] = self.log_prob(x_0, v_0)
        x[:,:,0], v[:,:,0] = x_0.clone(), v_0.clone()
        for i in range(1,n+1):
            x[:,:,i], v[:,:,i], log_p[:,0,i], _ = integrators.leap_frog(x[:,:,i-1], v[:,:,i-1], 
                                                                        self.eps, self.target, self.momentum)
        
        log_accept_p = log_p[:,0,1:] - log_p[:,0,0].unsqueeze(1)
        log_u = torch.log(torch.zeros_like(log_accept_p).uniform_())
        accepted_mask = (log_accept_p > log_u).float()
        last_accepted = accepted_mask[:,-1].unsqueeze(1)
        next_x = x[:,:,-1]*last_accepted + x[:,:,0]*(1-last_accepted)
        next_v = v[:,:,-1]*last_accepted + v[:,:,0]*(1-last_accepted)
        next_state = (next_x, next_v)
        
        samples = x[:,:,1:]*accepted_mask.unsqueeze(1)+x[:,:,:1].repeat([1,1,n])*(1-accepted_mask).unsqueeze(1)
        log_accept_p = torch.minimum(log_accept_p[:,-1], torch.zeros_like(log_accept_p[:,-1]))
        return samples, next_state, log_accept_p
    
    def sample_n(self, x_0, n_iterations, budget=None):
        assert (x_0.shape[1] == self.target.dim)
        
        batch_size = x_0.shape[0]
        all_samples = []
        acceptance_rates = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        trajectories_l = torch.zeros(n_iterations, dtype=torch.long, device=self.device)
        grad_evals = 0
        iterations_done = 0
        
        v_0 = self.momentum.sample(batch_size)
        x, v = x_0, v_0
        for i in range(n_iterations):
            n_steps, t = self.set_n_steps()
            trajectories_l[i] = n_steps+1
            v = np.sqrt(1-self.alpha)*v+np.sqrt(self.alpha)*self.momentum.sample(batch_size)
            samples, next_state, log_accept_p = self.iterate(x, v)
            all_samples.append(samples.cpu())
            x, v = next_state
            
            acceptance_rates += torch.exp(log_accept_p)
            grad_evals += n_steps
            iterations_done += 1
            if (budget is not None) and (budget < grad_evals):
                break
            
        all_samples = torch.cat(all_samples, dim=2)
        all_weights = torch.ones([batch_size, 1, all_samples.shape[2]], device=torch.device('cpu'))
        acceptance_rates /= iterations_done
        return all_samples, all_weights, acceptance_rates, grad_evals, trajectories_l
    
    def subsample(self, samples, weights, n):
        ids = torch.multinomial(weights.squeeze(1), n)
        ids, _ = torch.sort(ids, dim=1)
        ids = ids.unsqueeze(1).repeat([1,self.target.dim,1])
        return samples.gather(2, ids)
    

class oMCMC:
    def __init__(self, target, trajectory_length, step_size, gamma=0.0, sigma=1.0, alpha=1.0, jitter=False):
        self.device = target.device
        self.momentum = distributions.Normal(torch.zeros(target.dim, device=self.device), sigma, self.device)
        self.target = target
        self.eps = step_size
        self.gamma = gamma
        self.T_max = trajectory_length
        self.alpha = alpha
        self.jitter = jitter
        
    def set_n_steps(self):
        T = self.T_max
        if self.jitter:
            T = torch.ones([1], device=self.device).uniform_()*T
        self.n_steps = torch.ceil(T/self.eps).long()
        return self.n_steps
        
    def max_n_steps(self):
        n_steps = torch.ceil(self.T_max/self.eps).long()
        return n_steps
    
    @torch.no_grad()
    def log_prob(self, x, v):
        return self.target.log_prob(x) + self.momentum.log_prob(v)
        
    def iterate(self, x_0, v_0, d_0):
        batch_size = x_0.shape[0]
        batch_ids = torch.arange(batch_size, device=self.device)
        n = self.n_steps[0]
        
        x = torch.zeros([batch_size, self.target.dim, n + 1], device=self.device)
        v = torch.zeros([batch_size, self.target.dim, n + 1], device=self.device)
        log_p = torch.zeros([batch_size, 1, n + 1], device=self.device)
        log_det = torch.zeros([batch_size, 1, n + 1], device=self.device)
        d = torch.zeros([batch_size, 1, n + 1], device=self.device, dtype=torch.long)
        directions = torch.ones([batch_size], device=self.device, dtype=torch.long)
        i_0 = d_0.clone().flatten()
        x[batch_ids,:,i_0], v[batch_ids,:,i_0], d[batch_ids,:,i_0] = x_0.clone(), v_0.clone(), d_0.clone()
        log_p[batch_ids,0,i_0] = self.log_prob(x[batch_ids,:,i_0],v[batch_ids,:,i_0])
        
        i_prev = i_0.clone()    
        for _ in range(n):
            i_next = i_prev + directions
            flip_ids = (i_next == n+1)
            directions[flip_ids] = -1
            i_prev[flip_ids] = i_0[flip_ids]
            i_next[flip_ids] = i_0[flip_ids]+directions[flip_ids]
            _x, _v, _log_p, _log_det = integrators.leap_frog(x[batch_ids,:,i_prev], v[batch_ids,:,i_prev], 
                                                             self.eps, self.target, self.momentum, self.gamma, directions)
            x[batch_ids,:,i_next], v[batch_ids,:,i_next], log_p[batch_ids,0,i_next] = _x.clone(), _v.clone(), _log_p.clone()
            log_det[batch_ids,:,i_next] = log_det[batch_ids,:,i_prev] + _log_det
            d[batch_ids,0,i_next] = d[batch_ids,0,i_prev]+directions
            i_prev = i_next
        
        samples = x
        log_p -= log_p.median()
        log_weights = log_p + log_det
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=2, keepdim=True))
        
        next_ids = torch.multinomial(weights.squeeze(1), 1).flatten()
        next_x = x[batch_ids,:,next_ids]
        next_v = v[batch_ids,:,next_ids]
        next_d = d[batch_ids,:,next_ids]
        next_state = (next_x, next_v, next_d)
        
        acceptance_prob = (1-weights[batch_ids,:,i_0])
        return samples, weights, next_state, acceptance_prob
    
    def sample_n(self, x_0, n_iterations, budget=None):
        assert (x_0.shape[1] == self.target.dim)
        batch_size = x_0.shape[0]
        all_samples = []
        all_weights = []
        acceptance_rates = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        trajectories_l = []
        grad_evals = 0
        iterations_done = 0
        
        v_0 = self.momentum.sample(batch_size)
        d_0 = torch.zeros([batch_size,1], device=self.device, dtype=torch.long)
        x, v, d = x_0, v_0, d_0
        for i in range(n_iterations):
            n_steps = self.set_n_steps()
            trajectories_l.append(n_steps+1)
            if i > 0:
                d = (d + self.n_steps//2 + 2) % (self.n_steps + 1)
                v = np.sqrt(1-self.alpha)*v+np.sqrt(self.alpha)*self.momentum.sample(1)
            samples, weights, next_state, acceptance_prob = self.iterate(x, v, d)
            x, v, d = next_state
            
            all_samples.append(samples.cpu())
            all_weights.append(weights.cpu())
            acceptance_rates += acceptance_prob.flatten().float()
            grad_evals += n_steps
            iterations_done += 1
            if (budget is not None) and (budget < grad_evals):
                break
        all_samples = torch.cat(all_samples, dim=2)
        all_weights = torch.cat(all_weights, dim=2)
        trajectories_l = torch.stack(trajectories_l, dim=0)
        acceptance_rates /= iterations_done
        return all_samples, all_weights, acceptance_rates, grad_evals, trajectories_l
    
    def subsample(self, samples, weights, n):
        ids = torch.multinomial(weights.squeeze(1), n)
        ids, _ = torch.sort(ids, dim=1)
        ids = ids.unsqueeze(1).repeat([1,self.target.dim,1])
        return samples.gather(2, ids)
    
    
class optMCMC:
    def __init__(self, target, step_size, beta=0.95, sigma=1.0, alpha=1.0):
        self.device = target.device
        self.momentum = distributions.Normal(torch.zeros(target.dim, device=self.device), sigma, self.device)
        self.target = target
        self.max_eps = step_size
        self.pos_beta = beta
        self.alpha = alpha
    
    @torch.no_grad()
    def log_prob(self, x, v):
        return self.target.log_prob(x) + self.momentum.log_prob(v)
    
    def set_n_steps(self):
        T = self.T_max
        if self.jitter:
            T = torch.ones([1], device=self.device).uniform_()*T
        self.n_steps = torch.ceil(T/self.eps).long()
        return self.n_steps
    
    def set_eps_beta(self, i):
        self.eps = self.max_eps
        self.beta = self.pos_beta
        
    def iterate(self, x_0, v_0):
        x = [x_0.clone()]
        v = [v_0.clone()]
        log_p = [self.log_prob(x[0],v[0]).reshape([1,1])]
        log_det = [torch.zeros([1,1], device=self.device, dtype=torch.float)]
        
        i = 0
        log_w_max = log_p[0]
        _log_w = log_p[0]
        while _log_w > log_w_max - np.log(1e3):
            _x, _v, _log_p, _log_det = integrators.leap_frog_forward(x[i], v[i], self.eps, self.target, 
                                                                     self.momentum, self.beta)
            x.append(_x) 
            v.append(_v)
            log_p.append(_log_p.reshape([1,1]))
            log_det.append(_log_det + log_det[i])
            _log_w = _log_p + _log_det + log_det[i]
            if _log_w > log_w_max:
                log_w_max = _log_w
            i += 1
            
        i_0 = 0
        _log_w = log_p[0]
        while _log_w > log_w_max - np.log(1e3):
            _x, _v, _log_p, _log_det = integrators.leap_frog_backward(x[0], v[0], self.eps, self.target, 
                                                                      self.momentum, self.beta)
            x.insert(0, _x) 
            v.insert(0, _v)
            log_p.insert(0, _log_p.reshape([1,1]))
            log_det.insert(0, _log_det + log_det[0])
            _log_w = _log_p + log_det[0]
            if _log_w > log_w_max:
                log_w_max = _log_w
            i_0 += 1
        x = torch.stack(x, dim=2)
        v = torch.stack(v, dim=2)
        log_p = torch.stack(log_p, dim=2)
        log_p -= log_p.max()
        log_det = torch.stack(log_det, dim=2)
        log_weights = log_p + log_det
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=2, keepdim=True))
        samples = x
        
        next_ids = torch.multinomial(weights.squeeze(1), 1).squeeze()
        next_x = x[:,:,next_ids]
        next_v = v[:,:,next_ids]
        next_state = (next_x, next_v)
        
        trajectory_length = torch.tensor(x.shape[2], device=self.device)
        jump_length = torch.sqrt(((next_x-x[:,:,i_0])**2).sum())
        return samples, weights, next_state, trajectory_length, jump_length
    
    def sample_n(self, x_0, n_iterations, budget=None):
        assert (x_0.shape[1] == self.target.dim) and (x_0.shape[0] == 1)
        all_samples = []
        all_weights = []
        average_lenth = 0.0
        trajectories_l = []
        grad_evals = 0
        iterations_done = 0
        
        v_0 = self.momentum.sample(1)
        x, v = x_0, v_0
        for i in range(n_iterations):
            self.set_eps_beta(i)
            samples, weights, next_state, trajectory_length, jump_length = self.iterate(x, v)
            x, v = next_state
            v = np.sqrt(1-self.alpha)*v+np.sqrt(self.alpha)*self.momentum.sample(1)
            
            all_samples.append(samples.cpu())
            all_weights.append(weights.cpu())
            
            average_lenth += jump_length
            trajectories_l.append(trajectory_length)
            grad_evals += trajectory_length-1
            iterations_done += 1
            if (budget is not None) and (budget < grad_evals):
                break
            
        all_samples = torch.cat(all_samples, dim=2)
        all_weights = torch.cat(all_weights, dim=2)
        trajectories_l = torch.stack(trajectories_l, dim=0)
        average_lenth /= iterations_done
        return all_samples, all_weights, average_lenth, grad_evals, trajectories_l
    
    def subsample(self, samples, weights, n):
        ids = torch.multinomial(weights.squeeze(1), n)
        ids, _ = torch.sort(ids, dim=1)
        ids = ids.unsqueeze(1).repeat([1,self.target.dim,1])
        return samples.gather(2, ids)

    
class optH:
    def __init__(self, target, step_size, gamma=1.0, sigma=1.0):
        self.device = target.device
        self.momentum = distributions.Normal(torch.zeros(target.dim, device=self.device), sigma, self.device)
        self.target = target
        self.eps = step_size
        self.gamma = gamma
    
    @torch.no_grad()
    def log_prob(self, x, v):
        return self.target.log_prob(x) + self.momentum.log_prob(v)
        
    def iterate(self, x_0, n):
        v_0 = torch.zeros_like(x_0)
#         v_0 = self.momentum.sample(1)
        x = [x_0.clone()]
        v = [v_0.clone()]
        log_p = [self.log_prob(x[0],v[0]).reshape([1,1])]
        log_det = [torch.zeros([1,1], device=self.device, dtype=torch.float)]
        log_w_0 = log_p[0]
        
        direction = torch.tensor([1], device=self.device)
        _log_w = log_p[0]
        for i in range(n):
            _x, _v, _log_p, _log_det = integrators.euler(x[i], v[i], self.eps, 
                                                         self.target, self.momentum, self.gamma, direction)
            x.append(_x)
            v.append(_v)
            log_p.append(_log_p.reshape([1,1]))
            log_det.append(_log_det + log_det[i])
            _log_w = _log_p + _log_det + log_det[i]
        x = torch.stack(x, dim=2)
        v = torch.stack(v, dim=2)
        log_p = torch.stack(log_p, dim=2)
        log_p -= log_p.median()
        log_det = torch.stack(log_det, dim=2)
        log_weights = log_p+log_det
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=2, keepdim=True))
        samples = x
        return samples, log_p, log_det, weights

    def subsample(self, samples, weights, n):
        ids = torch.multinomial(weights.squeeze(1), n)
        ids, _ = torch.sort(ids, dim=1)
        ids = ids.unsqueeze(1).repeat([1,self.target.dim,1])
        return samples.gather(2, ids)
