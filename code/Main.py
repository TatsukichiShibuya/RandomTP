from utils import worker_init_fn, set_seed, combined_loss, set_wandb, set_device
from dataset import make_MNIST

from models.bp_net import bp_net
from models.tp_net import tp_net

import os
import sys
import wandb
import torch
import argparse
import numpy as np
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
BP_LIST = ["BP"]
TP_LIST = ["TP"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=str, default="MNIST", choices=["MNIST"])
    parser.add_argument("--algorithm", type=str, default="BP", choices=BP_LIST + TP_LIST)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)

    # model architecture
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--direct_depth", type=int, default=1)
    parser.add_argument("--in_dim", type=int, default=784)
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=10)

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-2)
    parser.add_argument("--learning_rate_backward", "-lrb", type=float, default=1e-2)
    parser.add_argument("--stepsize", type=float, default=1e-2)

    parser.add_argument("--label_augmentation", action="store_true")

    # setting of tp_layer
    parser.add_argument("--forward_function_1", "-ff1", type=str, default="identity",
                        choices=["identity", "random", "parameterized"])
    parser.add_argument("--forward_function_2", "-ff2", type=str, default="identity",
                        choices=["identity", "random", "parameterized"])
    parser.add_argument("--backward_function_1", "-bf1", type=str, default="identity",
                        choices=["identity", "random", "parameterized"])
    parser.add_argument("--backward_function_2", "-bf2", type=str, default="identity",
                        choices=["identity", "random", "difference"])

    # neccesary if {parameterized, random} was choosed
    parser.add_argument("--forward_function_1_init", "-ff1_init", type=str, default="orthogonal",
                        choices=["orthogonal", "gaussian", "uniform"])
    parser.add_argument("--forward_function_2_init", "-ff2_init", type=str, default="orthogonal",
                        choices=["orthogonal", "gaussian", "uniform"])
    parser.add_argument("--backward_function_1_init", "-bf1_init", type=str, default="orthogonal",
                        choices=["orthogonal", "gaussian", "uniform"])
    parser.add_argument("--backward_function_2_init", "-bf2_init", type=str, default="orthogonal",
                        choices=["orthogonal", "gaussian", "uniform"])

    # wandb
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--agent", action="store_true")

    args = parser.parse_args()
    return args


def main(**kwargs):
    set_seed(kwargs["seed"])
    set_wandb(**kwargs)
    device = set_device()
    print(f"DEVICE: {device}")

    if kwargs["problem"] == "MNIST":
        num_classes = 10
        trainset, validset, testset = make_MNIST(kwargs["label_augmentation"], kwargs["out_dim"])
    else:
        raise NotImplementedError()

    if kwargs["label_augmentation"]:
        loss_function = (lambda pred, label: combined_loss(pred, label, device, num_classes))
    else:
        loss_function = nn.CrossEntropyLoss(reduction="sum")

    # make dataloader
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=kwargs["batch_size"],
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=kwargs["batch_size"],
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=kwargs["batch_size"],
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True,
                                              worker_init_fn=worker_init_fn)

    # initialize model
    if kwargs["algorithm"] in BP_LIST:
        model = bp_net(depth=kwargs["depth"],
                       in_dim=kwargs["in_dim"],
                       out_dim=kwargs["out_dim"],
                       hid_dim=kwargs["hid_dim"],
                       activation_function=kwargs["activation_function"],
                       loss_function=loss_function,
                       device=device)
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["learning_rate"],
                    kwargs["log"])
    elif kwargs["algorithm"] in TP_LIST:
        params = set_params(kwargs)
        model = tp_net(kwargs["depth"], kwargs["direct_depth"], kwargs["in_dim"],
                       kwargs["hid_dim"], kwargs["out_dim"], loss_function, device, params=params)
        model.train(train_loader, valid_loader, kwargs["epochs"], kwargs["learning_rate"],
                    kwargs["learning_rate_backward"], kwargs["stepsize"], kwargs["log"])

    # test
    loss, acc = model.test(test_loader)
    print(f"Test Loss      : {loss}")
    if acc is not None:
        print(f"Test Acc       : {acc}")


def set_params(kwargs):
    params = {}
    long = ["forward_function_1", "forward_function_2",
            "backward_function_1", "backward_function_2"]
    short = ["ff1", "ff2", "bf1", "bf2"]
    for l, s in zip(long, short):
        if kwargs[l] in ["random", "parameterized"]:
            params[s] = {"type": kwargs[l], "init": kwargs[l + "_init"]}
        elif kwargs[l] in ["identity", "difference"]:
            params[s] = {"type": kwargs[l]}
    return params


if __name__ == '__main__':
    FLAGS = vars(get_args())
    # print(FLAGS)
    print(f"Forward  : {FLAGS['forward_function_1']} -> {FLAGS['forward_function_2']}")
    print(f"Backward : {FLAGS['backward_function_1']} -> {FLAGS['backward_function_2']}")

    main(**FLAGS)
