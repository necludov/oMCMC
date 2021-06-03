import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('..')
from core import samplers, distributions, experiments
from utils.logger import Logger
from utils import utils

import pickle

def main():
    logger = Logger('icg')
    device = torch.device('cuda')
    def saver(saving_dict, sampler_name):
        path = '../logs/icg_' + sampler_name + '.pt'
        torch.save(saving_dict, path)
    
    batch_size = 100
    adapt_size = 1000
    n_iterations = 1000
    seed = 0
    
    torch.manual_seed(seed)
    target = distributions.ICG(50, device)
    experiments.run_experiment(logger, seed, target, batch_size, adapt_size, n_iterations, saver)

if __name__ == '__main__':
    main()
