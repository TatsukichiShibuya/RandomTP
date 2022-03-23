import torch
from abc import ABCMeta, abstractmethod


class abstract_function(metaclass=ABCMeta):
    def __init__(self, in_dim, out_dim, layer, device):
        self.layer = layer
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    @abstractmethod
    def forward(self, input, original, update):
        raise NotImplementedError()

    @abstractmethod
    def update(self, lr):
        raise NotImplementedError()

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError()


class identity_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        self.weight = torch.eye(self.out_dim, self.in_dim, device=self.device)

    def forward(self, input, original, update):
        return input @ self.weight.T

    def update(self, lr):
        # Nothing to do
        return

    def zero_grad(self):
        # Nothing to do
        return
