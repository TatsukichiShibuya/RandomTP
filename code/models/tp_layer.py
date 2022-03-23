import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from models.my_functions import *


class tp_layer:
    def __init__(self, in_dim, out_dim, device, params):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        # set forward functions
        self.forward_function_1 = identity_function(in_dim, in_dim,  self, self.device, params)
        self.forward_function_2 = identity_function(in_dim, out_dim, self, self.device, params)

        # set backward functions
        self.backward_function_1 = identity_function(out_dim, out_dim, self, self.device, params)
        self.backward_function_2 = identity_function(out_dim, in_dim, self, self.device, params)

        # values
        self.input = None
        self.hidden = None
        self.output = None
        self.target = None

    def forward(self, x, update=True):
        if update:
            self.input = x
            self.hidden = self.forward_function_1.forward(self.input, self.input, update=update)
            self.output = self.forward_function_2.forward(self.hidden, self.input, update=update)
            self.output = self.output.requires_grad_()
            self.output.retain_grad()
            return self.output
        else:
            h = self.forward_function_1.forward(x, x, update=update)
            y = self.forward_function_2.forward(h, x, update=update)
            return y

    def backward(self, x, update=True):
        h = self.backward_function_1.forward(x, x, update=update)
        y = self.backward_function_2.forward(h, x, update=update)
        return y

    def update_forward(self, lr):
        self.forward_function_1.update(lr)
        self.forward_function_2.update(lr)

    def update_backward(self, lr):
        self.forward_function_1.update(lr)
        self.forward_function_2.update(lr)

    def zero_grad(self):
        if self.output.grad is not None:
            self.output.grad.zero_()
        self.forward_function_1.zero_grad()
        self.forward_function_2.zero_grad()
        self.backward_function_1.zero_grad()
        self.backward_function_2.zero_grad()
