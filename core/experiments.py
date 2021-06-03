import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('.')
from core import samplers, distributions
from utils.logger import Logger
from utils import utils

import pickle

def estimate_statistics(all_samples, all_weights, trajectories_l, name):
    all_samples = all_samples.cpu()
    all_weights = all_weights.cpu()
    trajectories_l = trajectories_l.cpu()
    device = torch.device('cpu')
    batch_size = all_samples.shape[0]
    d = all_samples.shape[1]
    if 'rechmc' == name:
        lengths = torch.cumsum(trajectories_l-1, 0)
    else:
        lengths = torch.cumsum(trajectories_l, 0)
    mean_estimates = torch.zeros([batch_size, d, len(trajectories_l)], device=device)
    variance_estimates = torch.zeros([batch_size, d, len(trajectories_l)], device=device)
    for i in range(len(lengths)):
        if 'hmc' == name:
            l = i+1
        else:
            l = lengths[i]
        mean = (all_samples[:,:,:l]*all_weights[:,:,:l]).sum(2)/all_weights[:,:,:l].sum(2)
        variance = (all_weights[:,:,:l]*(all_samples[:,:,:l]-mean.unsqueeze(2))**2).sum(2)/all_weights[:,:,:l].sum(2)
        mean_estimates[:,:,i] = mean
        variance_estimates[:,:,i] = variance
    grad_evals = torch.cumsum(trajectories_l-1, 0)
    return mean_estimates, variance_estimates, grad_evals

def run_chain(logger, chain, x_0, n_iterations, sample_size, given_budget, adapt_budget, name):
    logger.print('Running %s ...' % name)
    samples, weights, ar, budget, trajectories_l = chain.sample_n(x_0, n_iterations, given_budget)
    ess = []
    for _ in range(10):
        subsamples = chain.subsample(samples, weights, sample_size)
        ess.append(utils.batch_means_ess(subsamples.cpu().numpy()))
    ess = np.concatenate(ess, axis=0)
    logger.print('%s results:' % name)
    logger.print('mean AR = {:.3f}; min AR = {:.3f}; \
max AR = {:.3f}'.format(ar.mean().item(), ar.min().item(), ar.max().item()))
    logger.print('min ESS = {:.4e}; min ESS / budget = {:.4e}; \
budget = {}'.format(np.median(np.min(ess, axis=1)), np.median(np.min(ess, axis=1))/(budget+adapt_budget).item(), budget.item()))
    mean_estimates, variance_estimates, grad_evals = estimate_statistics(samples, weights, trajectories_l, name)
    return mean_estimates, variance_estimates, grad_evals, ess

def run_optmc(logger, chain, x_0, n_iterations, sample_size, given_budget, adapt_budget, name):
    logger.print('Running %s ...' % name)
    batch_size = x_0.shape[0]
    ess = []
    mean_estimates = []
    variance_estimates = []
    grad_evals = []
    for i in range(batch_size):
        logger.print('iteration {}/{}'.format(i, batch_size))
        samples, weights, ar, budget, trajectories_l = chain.sample_n(x_0[i:i+1], n_iterations, given_budget)
        subsamples = chain.subsample(samples, weights, sample_size)
        _ess = utils.batch_means_ess(subsamples.cpu().numpy())
        ess.append(utils.batch_means_ess(subsamples.cpu().numpy()))
        _mean, _variance, _evals = estimate_statistics(samples, weights, trajectories_l, name)
        mean_estimates.append(_mean)
        variance_estimates.append(_variance)
        grad_evals.append(_evals)
        logger.print('%s results:' % name)
        logger.print('average length = {:.4e}; average iter = {:.2f}; budget = {}'.format(ar.item(), trajectories_l.float().mean().cpu().item(), budget.item()))
        logger.print('min ESS = {:.4e}; min ESS / budget = {:.4e}; \
budget = {}'.format(np.min(_ess), np.min(_ess)/(budget+adapt_budget).item(), budget.item()))
    ess = np.concatenate(ess, axis=0)
    return mean_estimates, variance_estimates, grad_evals, ess

def run_experiment(logger, seed, target, batch_size, adapt_size, n_iterations, saver):
    logger.print('Starting experiment with seed={}'.format(seed))
    
    logger.print('Adapting HMC...')
    hmc = samplers.HMC(target)
    x_0 = torch.zeros([batch_size, target.dim], device=target.device).normal_()*1e-2
    x_0, adaptation_budget = hmc.adapt(x_0, adapt_size)
    logger.print('adaptation budget is {} with {} chains'.format(adaptation_budget.item(), batch_size))
    eps, T = hmc.eps, hmc.T_max
    logger.print('eps = {:.4f}; T = {:.4f}; T/eps = {}'.format(eps.cpu().numpy()[0], T.cpu().numpy()[0], 
                                                               torch.ceil(T/eps).cpu().numpy()[0]))
    
    hmc = samplers.HMC(target, T, eps)
    mean_estimates, variance_estimates, grad_evals, ess = run_chain(logger, hmc, x_0, n_iterations, 
                                                                    sample_size=n_iterations, given_budget=None, 
                                                                    adapt_budget=adaptation_budget, name='hmc')
    hmc_budget = grad_evals[-1].to(target.device)
    saving_dict = {'mean_estimates': mean_estimates, 
                   'variance_estimates': variance_estimates,
                   'grad_evals': grad_evals,
                   'ess': ess,
                   'adapt_budget': adaptation_budget}
    saver(saving_dict, 'hmc')
    del saving_dict, mean_estimates, variance_estimates
    
    rechmc = samplers.HMCRec(target, T, eps)
    mean_estimates, variance_estimates, grad_evals, ess = run_chain(logger, rechmc, x_0, n_iterations, 
                                                                    sample_size=n_iterations, given_budget=None, 
                                                                    adapt_budget=adaptation_budget, name='rechmc')
    saving_dict = {'mean_estimates': mean_estimates, 
                   'variance_estimates': variance_estimates,
                   'grad_evals': grad_evals,
                   'ess': ess,
                   'adapt_budget': adaptation_budget}
    saver(saving_dict, 'rechmc')
    del saving_dict, mean_estimates, variance_estimates
    
    max_iter = torch.ceil(T/eps).cpu().long().item()*n_iterations
    ohmc = samplers.oMCMC(target, T, eps)
    mean_estimates, variance_estimates, grad_evals, ess = run_chain(logger, ohmc, x_0, max_iter, 
                                                                    sample_size=n_iterations, 
                                                                    given_budget=hmc_budget, 
                                                                    adapt_budget=adaptation_budget, name='ohmc')
    saving_dict = {'mean_estimates': mean_estimates, 
                   'variance_estimates': variance_estimates,
                   'grad_evals': grad_evals,
                   'ess': ess,
                   'adapt_budget': adaptation_budget}
    saver(saving_dict, 'ohmc')
    del saving_dict, mean_estimates, variance_estimates
        
    optmc = samplers.optMCMC(target, eps, beta=0.8**(1/target.dim))
    mean_estimates, variance_estimates, grad_evals, ess = run_optmc(logger, optmc, x_0, max_iter, 
                                                                    sample_size=n_iterations, 
                                                                    given_budget=hmc_budget, 
                                                                    adapt_budget=adaptation_budget, name='optmc')
    saving_dict = {'mean_estimates': mean_estimates, 
                   'variance_estimates': variance_estimates,
                   'grad_evals': grad_evals,
                   'ess': ess,
                   'adapt_budget': adaptation_budget}
    saver(saving_dict, 'optmc')
    del saving_dict, mean_estimates, variance_estimates