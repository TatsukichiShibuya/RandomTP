import torch
from torch import nn
from abc import ABCMeta, abstractmethod


class abstract_function(metaclass=ABCMeta):
    def __init__(self, in_dim, out_dim, layer, device):
        self.layer = layer
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    @abstractmethod
    def forward(self, input, original=None):
        raise NotImplementedError()

    def update(self, lr):
        # Nothing to do
        return

    def zero_grad(self):
        # Nothing to do
        return


class identity_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        self.weight = torch.eye(out_dim, in_dim, device=device)

    def forward(self, input, original=None):
        return input @ self.weight.T


class parameterized_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        self.weight = torch.empty(out_dim, in_dim, requires_grad=True, device=device)
        if params["init"] == "uniform":
            nn.init.uniform_(self.weight, -1e-3, 1e-3)
        elif params["init"] == "gaussian":
            nn.init.normal_(self.weight, 0, 1e-3)
        elif params["init"] == "orthogonal":
            nn.init.orthogonal_(self.weight)
        else:
            raise NotImplementedError()
        self.activation_function = nn.Tanh()

    def forward(self, input, original=None):
        return self.activation_function(input @ self.weight.T)

    def update(self, lr):
        self.weight = (self.weight - lr * self.weight.grad).detach().requires_grad_()

    def zero_grad(self):
        if self.weight.grad is not None:
            self.weight.grad.zero_()


class random_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)
        self.weight = torch.empty(out_dim, in_dim, device=device)
        if params["init"] == "uniform":
            nn.init.uniform_(self.weight, -1e-3, 1e-3)
        elif params["init"] == "gaussian":
            nn.init.normal_(self.weight, 0, 1e-3)
        elif params["init"] == "orthogonal":
            nn.init.orthogonal_(self.weight)
        else:
            raise NotImplementedError()
        self.activation_function = nn.Tanh()

    def forward(self, input, original=None):
        return self.activation_function(input @ self.weight.T)


class difference_function(abstract_function):
    def __init__(self, in_dim, out_dim, layer, device, params):
        super().__init__(in_dim, out_dim, layer, device)

    def forward(self, input, original=None):
        with torch.no_grad():
            upper = self.layer.forward(original, update=False)
            rec = self.layer.backward_function_1.forward(upper)
            difference = original - rec
        return input + difference
