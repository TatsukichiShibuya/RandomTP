import pdb
import os
import sys

for arg in sys.argv[1:]:
    if 'dataset' in arg:
        dataset = arg.split('=')[-1]


args = ' '.join(map(str, sys.argv[1:]))
option = ""
if dataset == "MNIST":
    option = " --in_dim=784 --hid_dim=256 --out_dim=10 --depth=6"
elif dataset == "FashionMNIST":
    option = " --in_dim=784 --hid_dim=256 --out_dim=10 --depth=6"
elif dataset == "CIFAR10":
    option = " --in_dim=3072 --hid_dim=256 --out_dim=10 --depth=4"
elif dataset == "CIFAR100":
    option = " --in_dim=3072 --hid_dim=1024 --out_dim=100 --depth=4"

command = f'python Main.py {args} --log --agent' + option
print(command)
os.system(command)
