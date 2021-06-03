import torch
import numpy as np
import matplotlib.pyplot as plt

def get_errors(filename, target, sampler_name, grid):
    sampler_dict = torch.load(filename, map_location=target.device)
    mean_estimates = sampler_dict['mean_estimates']
    variance_estimates = sampler_dict['variance_estimates']
    ess = np.min(sampler_dict['ess'], axis=1)
    grad_evals = sampler_dict['grad_evals'].numpy().astype(np.float)
    budget = grad_evals[-1]
    adapt_budget = sampler_dict['adapt_budget'].numpy()
    print('{}: ESS = {:.2e}({:.2e}); ESS/budget = {:.2e}({:.2e});  \
budget = {}'.format(sampler_name, np.median(ess), np.std(ess), np.median(ess)/(budget+adapt_budget).item(), 
                        np.std(ess)/budget.item(), budget.item()))
    error_mean = np.abs(mean_estimates.numpy()-target.mean()[np.newaxis,:,np.newaxis]).sum(1)
    error_std = np.abs(np.sqrt(variance_estimates.numpy())-target.std()[np.newaxis,:,np.newaxis]).sum(1)
    aligned_mean = np.zeros([error_mean.shape[0], len(grid)])
    aligned_std = np.zeros([error_mean.shape[0], len(grid)])
    for i in range(error_mean.shape[0]):
        aligned_mean[i] = np.interp(grid, grad_evals.flatten(), error_mean[i])
        aligned_std[i] = np.interp(grid, grad_evals.flatten(), error_std[i])
    return aligned_mean, aligned_std

def get_errors_opt(filename, target, sampler_name, grid):
    sampler_dict = torch.load(filename, map_location=target.device)
    mean_estimates = sampler_dict['mean_estimates']
    variance_estimates = sampler_dict['variance_estimates']
    ess = np.min(sampler_dict['ess'], axis=1)
    grad_evals = sampler_dict['grad_evals']
    adapt_budget = sampler_dict['adapt_budget'].numpy()
    budget = grad_evals[-1][-1]
    print('{}: ESS = {:.2e}({:.2e}); ESS/budget = {:.2e}({:.2e});  \
budget = {}'.format(sampler_name, np.median(ess), np.std(ess), np.median(ess)/(budget+adapt_budget).item(), 
                        np.std(ess)/budget.item(), budget.item()))
    error_mean = np.zeros([len(mean_estimates), len(grid)])
    error_std = np.zeros([len(mean_estimates), len(grid)])
    for i in range(len(mean_estimates)):
        old_grid = grad_evals[i].numpy().astype(float)
        mean = np.abs(mean_estimates[i].numpy()-target.mean()[np.newaxis,:,np.newaxis]).sum(1)
        error_mean[i] = np.interp(grid, old_grid.flatten(), mean.flatten())
        std = np.abs(np.sqrt(variance_estimates[i].numpy())-target.std()[np.newaxis,:,np.newaxis]).sum(1)
        error_std[i] = np.interp(grid, old_grid.flatten(), std.flatten())
    return error_mean, error_std

def get_final_estimate(filename, device):
    dictionary = torch.load(filename, map_location=device)
    mean = dictionary['mean_estimates'][:,:,-1].mean(0)
    std = torch.sqrt(dictionary['variance_estimates'][:,:,-1].mean(0))
    return mean, std

def plot_errors(errors, n_iterations, label):
    errors_sorted = np.sort(errors, axis=0)
    plt.plot(n_iterations, np.mean(errors, axis=0), lw=5, label=label)
    n_chains = errors.shape[0]
    plt.fill_between(n_iterations, errors_sorted[int(n_chains*0.25),:], 
                     errors_sorted[int(n_chains*0.75),:], alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
