from net import net
from fa_layer import fa_layer

import time
import wandb
import torch
import numpy as np
from tqdm import tqdm


class fa_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_layers(self, in_dim, hid_dim, out_dim, activation_function):
        layers = [None] * self.depth

        # first layer
        layers[0] = fa_layer(in_dim, hid_dim, activation_function, self.device, 0)
        # hidden layers
        for d in range(1, self.depth - 1):
            layers[d] = fa_layer(hid_dim, hid_dim, activation_function, self.device, d)
        # last layer
        layers[-1] = fa_layer(hid_dim, out_dim, activation_function, self.device, self.depth - 1)

        return layers

    def train(self, train_loader, valid_loader, epochs, lr, log):
        for e in range(epochs):
            start_time = time.time()

            # train forward
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.forward(x)

                self.update_weights(x, y, lr)

            end_time = time.time()
            print(f"epochs {e}: {end_time - start_time:.2f}")

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)

                if log:
                    # results
                    log_dict = {"train loss": train_loss,
                                "valid loss": valid_loss}
                    if train_acc is not None:
                        log_dict["train accuracy"] = train_acc
                    if valid_acc is not None:
                        log_dict["valid accuracy"] = valid_acc
                    log_dict["time"] = end_time - start_time

                    wandb.log(log_dict)
                else:
                    # results
                    print(f"\ttrain loss     : {train_loss}")
                    print(f"\tvalid loss     : {valid_loss}")
                    if train_acc is not None:
                        print(f"\ttrain acc      : {train_acc}")
                    if valid_acc is not None:
                        print(f"\tvalid acc      : {valid_acc}")

    def update_weights(self, x, y, lr):
        y_pred = self.forward(x).requires_grad_()
        y_pred.retain_grad()
        loss = self.loss_function(y_pred, y)
        batch_size = len(y)
        self.zero_grad()
        loss.backward()
        g = y_pred.grad
        grad = [None] * self.depth
        with torch.no_grad():
            for d in reversed(range(self.depth)):
                std = torch.std(self.layers[d].activation)
                g = g * self.layers[d].activation_derivative(self.layers[d].linear_activation) / std
                grad[d] = g.T @ self.layers[d - 1].linear_activation if d >= 1 else g.T @ x
                g = g @ self.layers[d].fixed_weight
        for d in range(self.depth):
            self.layers[d].weight = (self.layers[d].weight - (lr / batch_size)
                                     * grad[d]).detach().requires_grad_()

    def zero_grad(self):
        for d in range(self.depth):
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
