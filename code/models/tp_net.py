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
        params_last["ff2"]["act"] = "linear"
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
                self.train_back_weights(x, y, lrb, std)
        rec_loss = self.reconstruction_loss_of_dataset(train_loader)
        print(f"> {rec_loss}")

        # train forward
        for e in range(epochs):
            print(f"Epoch {e}")
            torch.cuda.empty_cache()

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.train_back_weights(x, y, lrb, std)

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                for i in range(5):
                    self.train_back_weights(x, y, lrb, std)
                self.compute_target(x, y, stepsize)
                self.update_weights(x, lr)

            """
            epsilon_l2_loss_sum = [torch.zeros(1) for d in range(self.depth)]
            epsilon_angle_loss_sum = [torch.zeros(1) for d in range(self.depth)]
            eta_l2_loss_sum = [torch.zeros(1) for d in range(self.depth)]
            with torch.no_grad():
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    epsilon_l2_loss, epsilon_angle_loss = self.compute_epsilon_loss(x)
                    eta_l2_loss = self.compute_eta_loss(x)
                    for d in range(self.depth):
                        epsilon_l2_loss_sum[d] += epsilon_l2_loss[d].sum()
                        epsilon_angle_loss_sum[d] += epsilon_angle_loss[d].sum()
                        eta_l2_loss_sum[d] += eta_l2_loss[d].sum()
                for d in range(self.depth):
                    epsilon_l2_loss_sum[d] /= len(train_loader.dataset)
                    epsilon_angle_loss_sum[d] /= len(train_loader.dataset)
                    eta_l2_loss_sum[d] /= len(train_loader.dataset)
            """

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                # rec_loss = self.reconstruction_loss_of_dataset(train_loader)
                layerwise_rec_loss = self.layerwise_reconstruction_loss_of_dataset(train_loader)
            # log
            if log:
                log_dict = {}
                log_dict["train loss"] = train_loss
                log_dict["valid loss"] = valid_loss
                if train_acc is not None:
                    log_dict["train accuracy"] = train_acc
                if valid_acc is not None:
                    log_dict["valid accuracy"] = valid_acc

                # log_dict["rec loss"] = rec_loss
                for d in range(len(layerwise_rec_loss)):
                    log_dict["rec loss " + str(d + 1)] = layerwise_rec_loss[d]
                """
                for d in range(1, self.depth - self.direct_depth + 1):
                    log_dict[f"epsilon l2 {d}"] = epsilon_l2_loss_sum[d].item()
                    log_dict[f"epsilon angle {d}"] = epsilon_angle_loss_sum[d].item()
                    log_dict[f"eta l2 {d}"] = eta_l2_loss_sum[d].item()
                """

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
                for d in range(1, self.depth - self.direct_depth + 1):
                    print(f"\tEpsilon-l2-{d}     : {epsilon_l2_loss_sum[d].item()}")
                    print(f"\tEpsilon-angle-{d}  : {epsilon_angle_loss_sum[d].item()}")
                    print(f"\tEta-l2-{d}         : {eta_l2_loss_sum[d].item()}")
                """

    def train_back_weights(self, x, y, lrb, std, loss_type="DTP"):
        if not self.back_trainable:
            return

        self.forward(x)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            if loss_type == "DTP":
                q = self.layers[d - 1].output.detach().clone()
                q = q + torch.normal(0, std, size=q.shape, device=self.device)
                q_upper = self.layers[d].forward(q)
                h = self.layers[d].backward_function_1.forward(q_upper)
            elif loss_type == "DRL":
                raise NotImplementedError()
            elif loss_type == "L-DRL":
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            loss = self.MSELoss(h, q)
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            if d == self.depth - self.direct_depth:
                self.layers[d].update_backward(lrb * 10 / len(x))
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

    def layerwise_reconstruction_loss(self, x):
        layerwise_rec_loss = torch.zeros(self.depth - self.direct_depth, device=self.device)
        self.forward(x)
        for d in range(1, self.depth - self.direct_depth + 1):
            plane = self.layers[d].backward_function_1.forward(self.layers[d].output)
            # diff = self.layers[d].backward_function_2.forward(plane, self.layers[d - 1].output)
            layerwise_rec_loss[d - 1] = self.MSELoss(self.layers[d - 1].output, plane)
        return layerwise_rec_loss

    def layerwise_reconstruction_loss_of_dataset(self, data_loader):
        layerwise_rec_loss = torch.zeros(self.depth - self.direct_depth, device=self.device)
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            layerwise_rec_loss = layerwise_rec_loss + self.layerwise_reconstruction_loss(x)
        if torch.isnan(layerwise_rec_loss).any():
            print("ERROR: rec loss diverged")
            sys.exit(1)
        return layerwise_rec_loss / len(data_loader.dataset)

    def compute_epsilon_loss(self, x):
        l2_loss = [torch.zeros(1) for d in range(self.depth)]
        angle_loss = [torch.zeros(1) for d in range(self.depth)]
        self.forward(x)
        for d in range(1, self.depth - self.direct_depth + 1):
            h = self.layers[d - 1].output.detach().clone()
            h_e = h + torch.normal(0, 0.03, size=h.shape, device=self.device)
            h_e_next = self.layers[d].forward(h_e, update=False)
            h_e_rec = self.layers[d].backward_function_1.forward(h_e_next)
            h_e_rec = self.layers[d].backward_function_2.forward(h_e_rec, h)
            l2_loss[d] = torch.norm(h_e - h_e_rec, dim=1)
            angle_loss[d] = calc_angle(h_e - h, h_e_rec - h)
        return l2_loss, angle_loss

    def compute_eta_loss(self, x):
        l2_loss = [torch.zeros(1) for d in range(self.depth)]
        self.forward(x)
        for d in range(1, self.depth - self.direct_depth + 1):
            h = self.layers[d - 1].output.detach().clone()
            h_next = self.layers[d].forward(h, update=False)
            h_next_e = h_next + torch.normal(0, 0.03, size=h_next.shape, device=self.device)
            h_e_rec = self.layers[d].backward_function_1.forward(h_next_e)
            h_e_rec = self.layers[d].backward_function_2.forward(h_e_rec, h)
            l2_loss[d] = torch.norm(h - h_e_rec, dim=1)
        return l2_loss
