import numpy as np
import torch

import sys
sys.path.append('../')
from utils import utils

def evaluate_grad(x, target):
    y = x.detach().clone()
    y.requires_grad = True
    log_p = target.log_prob(y)
    grad = torch.autograd.grad(log_p, y, torch.ones([y.shape[0]]).to(target.device))[0]
    assert (grad.shape[0] == x.shape[0]) and (grad.shape[1] == target.dim)
    return log_p.detach(), grad.detach()

def leap_frog(x_0, v_0, step_size, target, momentum, gamma=0.0, directions=None):
    batch_size = x_0.shape[0]
    eps = step_size.view([1])
    if directions is None:
        d = 1.0
    else:
        d = directions.unsqueeze(1)
    alpha = torch.exp(-d*gamma*0.5*eps) # contraction term
    log_p_x_0, grad_x_0 = evaluate_grad(x_0, target)
    v_1 = alpha*(v_0 - d*0.5*eps*(-grad_x_0))
    
    log_p_v_1, grad_v_1 = evaluate_grad(v_1, momentum)
    x_1 = x_0 + d*0.5*eps*(alpha+alpha**-1)*(-grad_v_1)
    
    log_p_x_1, grad_x_1 = evaluate_grad(x_1, target)
    v_2 = alpha*(v_1 - d*0.5*eps*(-grad_x_1))
    
    with torch.no_grad():
        log_p_output = log_p_x_1 + momentum.log_prob(v_2)
    log_det_jacobian = -d*gamma*eps*target.dim
    return x_1, v_2, log_p_output, log_det_jacobian

# def leap_frog_forward(x_0, v_0, step_size, target, momentum, beta=1.0):
#     batch_size = x_0.shape[0]
#     eps = step_size.view([1])
#     log_p_v_0, grad_v_0 = evaluate_grad(v_0, momentum)
#     x_1 = x_0 + 0.5*eps*beta*(-grad_v_0)
    
#     log_p_x_1, grad_x_1 = evaluate_grad(x_1, target)
#     v_1 = beta**2*v_0 - beta*eps*(-grad_x_1)
    
#     log_p_v_1, grad_v_1 = evaluate_grad(v_1, momentum)
#     x_2 = x_1 + 0.5*eps/beta*(-grad_v_1)
    
#     with torch.no_grad():
#         log_p_output = target.log_prob(x_2) + log_p_v_1
#     log_det_jacobian = target.dim*torch.tensor([2*np.log(np.abs(beta))], device=target.device)
#     return x_2, v_1, log_p_output, log_det_jacobian

# def leap_frog_backward(x_0, v_0, step_size, target, momentum, beta=1.0):
#     batch_size = x_0.shape[0]
#     eps = step_size.view([1])
#     log_p_v_0, grad_v_0 = evaluate_grad(v_0, momentum)
#     x_1 = x_0 - 0.5*eps/beta*(-grad_v_0)
    
#     log_p_x_1, grad_x_1 = evaluate_grad(x_1, target)
#     v_1 = beta**-2*v_0 + eps/beta*(-grad_x_1)
    
#     log_p_v_1, grad_v_1 = evaluate_grad(v_1, momentum)
#     x_2 = x_1 - 0.5*eps*beta*(-grad_v_1)
    
#     with torch.no_grad():
#         log_p_output = target.log_prob(x_2) + log_p_v_1
#     log_det_jacobian = -target.dim*torch.tensor([2*np.log(np.abs(beta))], device=target.device)
#     return x_2, v_1, log_p_output, log_det_jacobian


def leap_frog_forward(x_0, v_0, step_size, target, momentum, beta=1.0):
    batch_size = x_0.shape[0]
    eps = step_size.view([1])
    log_p_x_0, grad_x_0 = evaluate_grad(x_0, target)
    v_1 = beta*(v_0 - 0.5*eps*(-grad_x_0))
    
    log_p_v_1, grad_v_1 = evaluate_grad(v_1, momentum)
    x_1 = x_0 + 0.5*eps*(1./beta+beta)*(-grad_v_1)
    
    log_p_x_1, grad_x_1 = evaluate_grad(x_1, target)
    v_2 = beta*(v_1 - 0.5*eps*(-grad_x_1))
    
    with torch.no_grad():
        log_p_output = log_p_x_1 + momentum.log_prob(v_2)
    log_det_jacobian = target.dim*torch.tensor([2*np.log(np.abs(beta))], device=target.device)
    return x_1, v_2, log_p_output, log_det_jacobian

def leap_frog_backward(x_0, v_0, step_size, target, momentum, beta=1.0):
    batch_size = x_0.shape[0]
    eps = step_size.view([1])
    log_p_x_0, grad_x_0 = evaluate_grad(x_0, target)
    v_1 = v_0/beta + 0.5*eps*(-grad_x_0)
    
    log_p_v_1, grad_v_1 = evaluate_grad(v_1, momentum)
    x_1 = x_0 - 0.5*eps*(1./beta+beta)*(-grad_v_1)
    
    log_p_x_1, grad_x_1 = evaluate_grad(x_1, target)
    v_2 = v_1/beta + 0.5*eps*(-grad_x_1)
    
    with torch.no_grad():
        log_p_output = log_p_x_1 + momentum.log_prob(v_2)
    log_det_jacobian = -target.dim*torch.tensor([2*np.log(np.abs(beta))], device=target.device)
    return x_1, v_2, log_p_output, log_det_jacobian

def euler_forward(x_0, v_0, step_size, target, momentum, beta):
    batch_size = x_0.shape[0]
    eps = step_size.view([1])
    log_p_x_0, grad_x_0 = evaluate_grad(x_0, target)
    v_1 = beta*v_0 - (1-beta)*eps*(-grad_x_0)
    
    log_p_v_1, grad_v_1 = evaluate_grad(v_1, momentum)
    x_1 = x_0 + eps*(-grad_v_1)
    
    with torch.no_grad():
        log_p_output = target.log_prob(x_1) + log_p_v_1
    log_det_jacobian = target.dim*torch.tensor([np.log(np.abs(beta))], device=target.device)
    return x_1, v_1, log_p_output, log_det_jacobian

def euler_backward(x_0, v_0, step_size, target, momentum, beta):
    batch_size = x_0.shape[0]
    eps = step_size.view([1])
    log_p_v_0, grad_v_0 = evaluate_grad(v_0, momentum)
    x_1 = x_0 - eps*(-grad_v_0)
    
    log_p_x_1, grad_x_1 = evaluate_grad(x_1, target)
    v_1 = 1./beta*v_0 + (1-beta)/beta*eps*(-grad_x_1)
    with torch.no_grad():
        log_p_output = log_p_x_1 + momentum.log_prob(v_1)
    log_det_jacobian = -target.dim*torch.tensor([np.log(np.abs(beta))], device=target.device)
    return x_1, v_1, log_p_output, log_det_jacobian
