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

#################################################################################
# Code from HAMILTORCH: https://github.com/AdamCobb/hamiltorch
#################################################################################

def flatten(model):
    return torch.cat([p.flatten() for p in model.parameters()])

def unflatten(model, flattened_params):
    if flattened_params.dim() != 1:
        raise ValueError('Expecting a 1d flattened_params')
    params_list = []
    i = 0
    for val in list(model.parameters()):
        length = val.nelement()
        param = flattened_params[i:i+length].view_as(val)
        params_list.append(param)
        i += length

    return params_list


def update_model_params_in_place(model, params):
    for weights, new_w in zip(model.parameters(), params):
        weights.data = new_w


# Edited from https://github.com/mariogeiger/hessian
def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    '''
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


#################################################################################
# Found here: https://gist.github.com/apaszke/4c8ead6f17a781d589f6655692e7f6f0
#################################################################################

import sys
import types
from collections import OrderedDict

PY2 = sys.version_info[0] == 2
_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}


### Had to add this for conv net
_new_methods = {'conv2d_forward','_forward_impl', '_check_input_dim', '_conv_forward'}


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()


# Function keeps looping and turning each module in the network to a function
def _make_functional(module, params_box, params_offset):
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    # Set dummy variable to bias_None to rename as flag if no bias
    if 'bias' in param_names and module._parameters['bias'] is None:
        param_names[-1] = 'bias_None' # Remove last name (hopefully bias) from list
    forward = type(module).forward.__func__ if PY2 else type(module).forward
    if type(module) == torch.nn.modules.container.Sequential:
        # Patch sequential model by replacing the forward method
        forward = Sequential_forward_patch
    if 'BatchNorm' in module.__class__.__name__:
        # Patch sequential model by replacing the forward method (hoping applies for all BNs need
        # to put this in tests)
        forward = bn_forward_patch

    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue   #If internal attributes skip
        setattr(self, name, attr)
    ### Had to add this for conv net (MY ADDITION)
    for name in dir(module):
        if name in _new_methods:
            if name == '_conv_forward': # Patch for pytorch 1.5.0+cu101
                setattr(self, name, types.MethodType(type(module)._conv_forward,self))
            if name == 'conv2d_forward':
                setattr(self, name, types.MethodType(type(module).conv2d_forward,self))
            if name == '_forward_impl':
                setattr(self, name, types.MethodType(type(module)._forward_impl,self))
            if name == '_check_input_dim': # Batch Norm
                # import pdb; pdb.set_trace()
                setattr(self, name, types.MethodType(type(module)._check_input_dim,self))

    child_params_offset = params_offset + num_params
    for name, child in module.named_children():
        child_params_offset, fchild = _make_functional(child, params_box, child_params_offset)
        self._modules[name] = fchild  # fchild is functional child
        setattr(self, name, fchild)
    def fmodule(*args, **kwargs):

        # Uncomment below if statement to step through (with 'n') assignment of parameters.
#         if params_box[0] is not None:
#             import pdb; pdb.set_trace()

        # If current layer has no bias, insert the corresponding None into params_box
        # with the params_offset ensuring the correct weight is applied to the right place.
        if 'bias_None' in param_names:
            params_box[0].insert(params_offset + 1, None)
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):

            # In order to deal with layers that have no bias:
            if name == 'bias_None':
                setattr(self, 'bias', None)
            else:
                setattr(self, name, param)
        # In the forward pass we receive a context object and a Tensor containing the
        # input; we must return a Tensor containing the output, and we can use the
        # context object to cache objects for use in the backward pass.

        # When running the kwargs no longer exist as they were put into params_box and therefore forward is just
        # forward(self, x), so I could comment **kwargs out
        return forward(self, *args) #, **kwargs)

    return child_params_offset, fmodule


def make_functional(module):
    params_box = [None]
    _, fmodule_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        params_box[0] = kwargs.pop('params') # if key is in the dictionary, remove it and return its value, else return default. If default is not given and key is not in the dictionary, a KeyError is raised.
        return fmodule_internal(*args, **kwargs)

    return fmodule

##### PATCH FOR nn.Sequential #####

def Sequential_forward_patch(self, input):
    # put at top of notebook nn.Sequential.forward = Sequential_forward_patch
    for label, module in self._modules.items():
        input = module(input)
    return input

##### Patch for batch norm #####
def bn_forward_patch(self, input):
    # set running var to None and running mean
    return torch.nn.functional.batch_norm(
                input, running_mean = None, running_var = None,
                weight = self.weight, bias = self.bias,
                training = self.training,
                momentum = self.momentum, eps = self.eps)

def gpu_check_delete(string, locals):
    if string in locals:
        del locals[string]
        torch.cuda.empty_cache()
