import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms


class MyClassification(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label


class MyAugmentedClassification(torch.utils.data.Dataset):
    def __init__(self, X, y, dim, num_classes):
        self.X = X
        self.y = F.one_hot(y, num_classes=num_classes)
        noize = torch.normal(0, 1, size=(len(X), dim - num_classes))
        self.y = torch.concat([self.y, noize], dim=1)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label


def make_MNIST(label_augmentation=False, dim=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([60000, 784]), torch.empty([60000], dtype=torch.long)
    for i, t in enumerate(list(mnist_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        trainset = MyAugmentedClassification(train_x, train_y, dim, 10)
    else:
        trainset = MyClassification(train_x, train_y)

    mnist_test = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 784]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(mnist_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        testset = MyAugmentedClassification(test_x, test_y, dim, 10)
    else:
        testset = MyClassification(test_x, test_y)

    return trainset, testset


def make_fashionMNIST(label_augmentation=False, dim=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    fashion_train = tv.datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
    train_x, train_y = torch.empty([60000, 784]), torch.empty([60000], dtype=torch.long)
    for i, t in enumerate(list(fashion_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        trainset = MyAugmentedClassification(train_x, train_y, dim, 10)
    else:
        trainset = MyClassification(train_x, train_y)

    fashion_test = tv.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 784]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(fashion_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        testset = MyAugmentedClassification(test_x, test_y, dim, 10)
    else:
        testset = MyClassification(test_x, test_y)

    return trainset, testset


def make_CIFAR10(label_augmentation=False, dim=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    cifar_train = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([50000, 3072]), torch.empty([50000], dtype=torch.long)
    for i, t in enumerate(list(cifar_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        trainset = MyAugmentedClassification(train_x, train_y, dim, 10)
    else:
        trainset = MyClassification(train_x, train_y)

    cifar_test = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 3072]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(cifar_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        testset = MyAugmentedClassification(test_x, test_y, dim, 10)
    else:
        testset = MyClassification(test_x, test_y)

    return trainset, testset


def make_CIFAR100(label_augmentation=False, dim=None):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    cifar_train = tv.datasets.CIFAR100(root='./data', train=True,
                                       download=True, transform=transform)
    train_x, train_y = torch.empty([50000, 3072]), torch.empty([50000], dtype=torch.long)
    for i, t in enumerate(list(cifar_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        trainset = MyAugmentedClassification(train_x, train_y, dim, 100)
    else:
        trainset = MyClassification(train_x, train_y)

    cifar_test = tv.datasets.CIFAR100(root='./data', train=False,
                                      download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 3072]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(cifar_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]
    if label_augmentation:
        testset = MyAugmentedClassification(test_x, test_y, dim, 100)
    else:
        testset = MyClassification(test_x, test_y)

    return trainset, testset
