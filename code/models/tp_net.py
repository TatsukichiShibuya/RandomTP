from models.tp_layer import tp_layer
from models.net import net

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

    def init_layers(self, in_dim, hid_dim, out_dim, params):
        layers = [None] * self.depth
        dims = [in_dim] + [hid_dim] * (self.depth - 1) + [out_dim]
        for d in range(self.depth):
            layers[d] = tp_layer(dims[d], dims[d + 1], self.device, params)
        return layers

    def forward(self, x, update=True):
        y = x
        for d in range(self.depth):
            y = self.layers[d].forward(y, update=update)
        return y

    def train(self, train_loader, valid_loader, epochs, lr, lrb, stepsize, log, params=None):
        # reconstruction loss
        rec_loss = self.reconstruction_loss_of_dataset(train_loader)
        print(f"\tRec Loss       : {rec_loss}")

        # train backward
        for e in range(5):
            print(f"Epoch {e}")
            torch.cuda.empty_cache()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.train_back_weights(x, y, lrb)

        # train forward
        for e in range(epochs):
            print(f"Epoch {e}")
            torch.cuda.empty_cache()

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.train_back_weights(x, y, lrb)

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.compute_target(x, y, stepsize)
                self.update_weights(x, lr)

            # predict
            with torch.no_grad():
                train_loss, train_acc = self.test(train_loader)
                valid_loss, valid_acc = self.test(valid_loader)
                rec_loss = self.reconstruction_loss_of_dataset(train_loader)

            # log
            if log:
                log_dict["train loss"] = train_loss
                log_dict["valid loss"] = valid_loss
                log_dict["rec loss"] = rec_loss
                if train_acc is not None:
                    log_dict["train accuracy"] = train_acc
                if valid_acc is not None:
                    log_dict["valid accuracy"] = valid_acc
                wandb.log(log_dict)
            else:
                print(f"\tTrain Loss     : {train_loss}")
                print(f"\tValid Loss     : {valid_loss}")
                print(f"\tRec Loss       : {rec_loss}")
                if train_acc is not None:
                    print(f"\tTrain Acc      : {train_acc}")
                if valid_acc is not None:
                    print(f"\tValid Acc      : {valid_acc}")

    def train_back_weights(self, x, y, lrb):
        return

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
                self.layers[d].target = self.layers[d + 1].backward(self.layers[d + 1].target)

    def update_weights(self, x, lr):
        self.forward(x)
        batch_size = len(x)
        for d in reversed(range(self.depth)):
            t, h = self.layers[d].target, self.layers[d].output
            loss = self.MSELoss(t, h)
            self.layers[d].zero_grad()
            loss.backward(retain_graph=True)
            self.layers[d].update_forward(lr / batch_size)

    def reconstruction_loss(self, x):
        h_bottom = self.layers[0].forward(x)
        h = h_bottom
        for d in range(1, self.depth - self.direct_depth + 1):
            h = self.layers[d].forward(h)
        for d in reversed(range(1, self.depth - self.direct_depth + 1)):
            h = self.layers[d].backward(h)
        return self.MSELoss(h_bottom, h)

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
        h = self.layers[0].forward(x)
        for d in range(1, self.depth - self.direct_depth + 1):
            h_upper = self.layers[d].forward(h, update=False)
            h_rec = self.layers[d].backward(h_upper)
            layerwise_loss_sum[d - 1] = self.MSELoss(h, h_rec)
            h = h_upper
        return layerwise_rec_loss

    def layerwise_reconstruction_loss_of_dataset(self, data_loader):
        layerwise_rec_loss = torch.zeros(self.depth - self.direct_depth, device=self.device)
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            layerwise_rec_loss = layerwise_rec_loss + layerwise_reconstruction_loss(x)
        if torch.isnan(rec_loss).any():
            print("ERROR: rec loss diverged")
            sys.exit(1)
        return layerwise_rec_loss / len(data_loader.dataset)
