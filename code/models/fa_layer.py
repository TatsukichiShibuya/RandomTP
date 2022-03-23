import torch
from torch import nn
from utils import batch_normalization, get_seed

import sys


class fa_layer:
    def __init__(self, in_dim, out_dim, activation_function, device, seed):
        # weights
        get_seed(seed, device)
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        nn.init.orthogonal_(self.weight)

        # fixed weight
        r = 1e-4
        shape = self.weight.shape
        gen = get_seed(seed, device)
        self.fixed_weight = torch.zeros(size=shape, device=device).uniform_(-r, r, generator=gen)

        # functions
        if activation_function == "leakyrelu":
            self.activation_function = nn.LeakyReLU(0.2)
        elif activation_function == "relu":
            self.activation_function = nn.LeakyReLU(0)
        elif activation_function == "sigmoid":
            self.activation_function = nn.Sigmoid()
        elif activation_function == "linear":
            self.activation_function = (lambda x: x)
            self.activation_derivative = (lambda x: 1)
        elif activation_function == "tanh":
            self.activation_function = nn.Tanh()
            self.activation_derivative = (lambda x: 1 - torch.tanh(x)**2)
        else:
            sys.tracebacklimit = 0
            raise NotImplementedError(f"activation_function : {activation_function} ?")

        # activation
        self.linear_activation = None
        self.activation = None

    def forward(self, x, update=True):
        if update:
            self.linear_activation = x @ self.weight.T
            self.activation = self.activation_function(self.linear_activation)
            #self.activation = batch_normalization(self.activation)
            return self.activation
        else:
            a = x @ self.weight.T
            h = self.activation_function(a)
            #h = batch_normalization(h)
            return h
