from models.tp_layer import tp_layer
from models.net import net
from models.my_functions import parameterized_function
from utils import calc_angle
from copy import deepcopy

import sys
import time
import wandb
import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian


class tp_net(net):
    def __init__(self, depth, direct_depth, in_dim, hid_dim, out_dim, loss_function, device, params=None):
        self.depth = depth
        self.direct_depth = direct_depth
        self.loss_function = loss_function
        self.device = device
        self.MSELoss = nn.MSELoss(reduction="sum")
        self.layers = self.init_layers(in_dim, hid_dim, out_dim, params)
        self.back_trainable = (params["bf1"]["type"] == "parameterized")

    def init_layers(self, in_dim, hid_dim, out_dim, params):
        layers = [None] * self.depth
        dims = [in_dim] + [hid_dim] * (self.depth - 1) + [out_dim]
        for d in range(self.depth - 1):
            layers[d] = tp_layer(dims[d], dims[d + 1], self.device, params)
        params_last = deepcopy(params)
        params_last["ff2"]["act"] = params["last"]
        layers[-1] = tp_layer(dims[-2], dims[-1], self.device, params_last)

        return layers

    def forward(self, x, update=True):
        y = x
        for d in range(self.depth):
            y = self.layers[d].forward(y, update=update)
        return y

    def train(self, train_loader, valid_loader, epochs, lr, lrb, std, stepsize, log, params=None):
        # reconstruction loss
        rec_loss = self.reconstruction_loss_of_dataset(train_loader)
        print(f"Initial Rec Loss: {rec_loss} ", end="")

        # train backward
        for e in range(5):
            print("-", end="")
            torch.cuda.empty_cache()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.train_back_weights(x, y, lrb, std, loss_type=params["loss_backward"])
        rec_loss = self.reconstruction_loss_of_dataset(train_loader)
        print(f"> {rec_loss}")

        # train forward
        for e in range(epochs + 1):
            print(f"Epoch {e}")
            torch.cuda.empty_cache()
            start_time = time.time()
            if e > 0 and params["epochs_backward"] > 0:
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.train_back_weights(x, y, lrb, std, loss_type=params["loss_backward"])

            # forward_loss_sum = [torch.zeros(1, device=self.device) for d in range(self.depth)]
            # target_rec_sum = [torch.zeros(1, device=self.device) for d in range(self.depth)]
            eigenvalues_sum = [torch.zeros(1, device=self.device) for d in range(self.depth)]

            if e > 0:
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    for i in range(params["epochs_backward"]):
                        self.train_back_weights(
                            x, y, lrb, std, loss_type=params["loss_backward"])
                    self.compute_target(x, y, stepsize)
                    self.update_weights(x, lr)

            if e == epochs or e == 0:
                with torch.no_grad():
                    self.forward(x)
                    for d in range(1, self.depth - self.direct_depth + 1):
                        h1 = self.layers[d].input[0]
                        gradf = jacobian(self.layers[d].forward, h1)
                        h2 = self.layers[d].forward(h1)
                        gradg = jacobian(self.layers[d].backward_function_1.forward, h2)
                        eig, _ = torch.linalg.eig(gradf @ gradg)
                        #eigenvalues_sum[d] += torch.trace(gradf @ gradg)
                        eigenvalues_sum[d] += (eig.real > 0).sum() / len(eig.real)
            for d in range(self.depth):
                eigenvalues_sum[d] /= len(train_loader)

            """
            with torch.no_grad():
                for d in range(1, self.depth - self.direct_depth + 1):
                    for i in range(5):
                        h1 = fixed_input[d][i]
                        gradf = jacobian(self.layers[d].forward, h1)
                        h2 = self.layers[d].forward(h1)
                        gradg = jacobian(self.layers[d].backward_function_1.forward, h2)
                        eigenvalues_sum[d] += torch.trace(gradf @ gradg)
                for d in range(self.depth):
                    eigenvalues_sum[d] /= 5
            """
            """
                with torch.no_grad():
                    for d in range(self.depth):
                        norm = torch.norm(self.layers[d].output - self.layers[d].target, dim=1)**2
                        norm /= (torch.norm(self.layers[d].output, dim=1) + 1e-12)
                        forward_loss_sum[d] += norm.sum()
                    for d in range(1, self.depth - self.direct_depth + 1):
                        rec = torch.norm(self.layers[d].target - self.layers[d].forward(self.layers[d - 1].target),
                                         dim=1)**2
                        rec /= (torch.norm(self.layers[d].target, dim=1)**2 + 1e-12)
                        target_rec_sum[d] += rec.sum()
            for d in range(self.depth):
                forward_loss_sum[d] /= len(train_loader.dataset)
                target_rec_sum[d] /= len(train_loader.dataset)
            """
            end_time = time.time()

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                layerwise_rec_loss = self.layerwise_reconstruction_loss_of_dataset(train_loader,
                                                                                   std, loss_type=params["loss_backward"])
            # log
            if log:
                log_dict = {}
                log_dict["train loss"] = train_loss
                log_dict["valid loss"] = valid_loss
                if train_acc is not None:
                    log_dict["train accuracy"] = train_acc
                if valid_acc is not None:
                    log_dict["valid accuracy"] = valid_acc
                log_dict["time"] = end_time - start_time

                for d in range(len(layerwise_rec_loss)):
                    log_dict["rec loss " + str(d + 1)] = layerwise_rec_loss[d]
                """
                for d in range(self.depth):
                    log_dict["forward loss " + str(d)] = forward_loss_sum[d]
                for d in range(1, self.depth - self.direct_depth + 1):
                    log_dict["target rec loss " + str(d)] = target_rec_sum[d]
                """
                for d in range(1, self.depth - self.direct_depth + 1):
                    log_dict[f"eigenvalue sum {d}"] = eigenvalues_sum[d].item()

                wandb.log(log_dict)
            else:
                print(f"\tTrain Loss       : {train_loss}")
                print(f"\tValid Loss       : {valid_loss}")
                if train_acc is not None:
                    print(f"\tTrain Acc        : {train_acc}")
                if valid_acc is not None:
                    print(f"\tValid Acc        : {valid_acc}")
                # print(f"\tRec Loss         : {rec_loss}")
                for d in range(len(layerwise_rec_loss)):
                    print(f"\tRec Loss-{d+1}       : {layerwise_rec_loss[d]}")
                """
                for d in range(self.depth):
                    print(f"\tForward Loss-{d}     : {forward_loss_sum[d].item()}")
                for d in range(1, self.depth - self.direct_depth + 1):
                    print(f"\tTarget Rec Loss-{d}  : {target_rec_sum[d].item()}")
                for d in range(1, self.depth - self.direct_depth + 1):
                    print(f"\teigenvalue sum-{d}: {eigenvalues_sum[d].item()}")
                """

    def train_back_weights(self, x, y, lrb, std, loss_type="L-DRL"):
        if not self.back_trainable:
            return

        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            if loss_type == "DTP":
                q = self.layers[d - 1].output.detach().clone()
                q = q + torch.normal(0, std, size=q.shape, device=self.device)
                q_upper = self.layers[d].forward(q)
                h = self.layers[d].backward_function_1.forward(q_upper)
                loss = self.MSELoss(h, q)
            elif loss_type == "DRL":
                h = self.layers[d - 1].output.detach().clone()
                q = h + torch.normal(0, std, size=h.shape, device=self.device)
                for _d in range(d, self.depth - self.direct_depth + 1):
                    q = self.layers[_d].forward(q)
                for _d in range(d, self.depth - self.direct_depth + 1):
                    h = self.layers[_d].forward(h)
                for _d in reversed(range(d, self.depth - self.direct_depth + 1)):
                    q = self.layers[_d].backward_function_1.forward(q)
                    q = self.layers[_d].backward_function_2.forward(q, self.layers[_d - 1].output)
                loss = self.MSELoss(self.layers[d].input.clone(), q)

            elif loss_type == "L-DRL":
                h = self.layers[d - 1].output.detach().clone()
                q = h + torch.normal(0, std, size=h.shape, device=self.device)
                q_up = self.layers[d].forward(q)
                _q_up = self.layers[d].backward_function_1.forward(q_up)
                q_rec = self.layers[d].backward_function_2.forward(_q_up, h)

                h_up = self.layers[d].forward(h)
                r_up = h_up + torch.normal(0, std, size=h_up.shape, device=self.device)
                _r_up = self.layers[d].backward_function_1.forward(r_up)
                r_rec = self.layers[d].backward_function_2.forward(_r_up, h)

                loss = -((q - h) * (q_rec - h)).sum() + self.MSELoss(r_rec, h) / 2
            else:
                raise NotImplementedError()
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            if d == self.depth - self.direct_depth:
                self.layers[d].update_backward(lrb / 10 / len(x))
            else:
                self.layers[d].update_backward(lrb / len(x))

    def compute_target(self, x, y, stepsize):
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y)
        for d in range(self.depth):
            self.layers[d].zero_grad()
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for d in range(self.depth - self.direct_depth, self.depth):
                self.layers[d].target = self.layers[d].output - \
                    stepsize * self.layers[d].output.grad

            for d in reversed(range(self.depth - self.direct_depth)):
                plane = self.layers[d + 1].backward_function_1.forward(self.layers[d + 1].target)
                diff = self.layers[d + 1].backward_function_2.forward(plane, self.layers[d].output)
                self.layers[d].target = diff

    def update_weights(self, x, lr):
        self.forward(x)
        for d in range(self.depth):
            loss = self.MSELoss(self.layers[d].target, self.layers[d].output)
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            self.layers[d].update_forward(lr / len(x))

    def reconstruction_loss(self, x):
        self.forward(x)
        h = self.layers[self.depth - self.direct_depth].output
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            plane = self.layers[d].backward_function_1.forward(h)
            diff = self.layers[d].backward_function_2.forward(plane, self.layers[d - 1].output)
            h = diff
        return self.MSELoss(self.layers[0].output, h)

    def reconstruction_loss_of_dataset(self, data_loader):
        rec_loss = 0
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            rec_loss = rec_loss + self.reconstruction_loss(x)
        if torch.isnan(rec_loss).any():
            print("ERROR: rec loss diverged")
            sys.exit(1)
        return rec_loss / len(data_loader.dataset)

    def layerwise_reconstruction_loss(self, x, std, loss_type):
        layerwise_rec_loss = torch.zeros(self.depth - self.direct_depth, device=self.device)
        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            if loss_type == "DTP":
                q = self.layers[d - 1].output.detach().clone()
                q = q + torch.normal(0, std, size=q.shape, device=self.device)
                q_upper = self.layers[d].forward(q)
                h = self.layers[d].backward_function_1.forward(q_upper)
                layerwise_rec_loss[d - 1] = self.MSELoss(h, q)
            elif loss_type == "DRL":
                h = self.layers[d - 1].output.detach().clone()
                q = h + torch.normal(0, std, size=h.shape, device=self.device)
                for _d in range(d, self.depth - self.direct_depth + 1):
                    q = self.layers[_d].forward(q)
                for _d in range(d, self.depth - self.direct_depth + 1):
                    h = self.layers[_d].forward(h)
                for _d in reversed(range(d, self.depth - self.direct_depth + 1)):
                    q = self.layers[_d].backward_function_1.forward(q)
                    q = self.layers[_d].backward_function_2.forward(q, self.layers[_d - 1].output)
                layerwise_rec_loss[d - 1] = self.MSELoss(self.layers[d].input.clone(), q)

            elif loss_type == "L-DRL":
                h = self.layers[d - 1].output.detach().clone()
                q = h + torch.normal(0, std, size=h.shape, device=self.device)
                q_up = self.layers[d].forward(q)
                _q_up = self.layers[d].backward_function_1.forward(q_up)
                q_rec = self.layers[d].backward_function_2.forward(_q_up, h)

                h_up = self.layers[d].forward(h)
                r_up = h_up + torch.normal(0, std, size=h_up.shape, device=self.device)
                _r_up = self.layers[d].backward_function_1.forward(r_up)
                r_rec = self.layers[d].backward_function_2.forward(_r_up, h)

                layerwise_rec_loss[d - 1] = - \
                    ((q - h) * (q_rec - h)).sum() + self.MSELoss(r_rec, h) / 2
            else:
                raise NotImplementedError()
        return layerwise_rec_loss

    def layerwise_reconstruction_loss_of_dataset(self, data_loader, std, loss_type):
        layerwise_rec_loss = torch.zeros(self.depth - self.direct_depth, device=self.device)
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            layerwise_rec_loss = layerwise_rec_loss + \
                self.layerwise_reconstruction_loss(x, std, loss_type=loss_type)
        if torch.isnan(layerwise_rec_loss).any():
            print("ERROR: rec loss diverged")
            sys.exit(1)
        return layerwise_rec_loss / len(data_loader.dataset)
