import os
import sys
args = ' '.join(map(str, sys.argv[1:]))
option = " --dataset=CIFAR100 --in_dim=3072 --hid_dim=631 --out_dim=100 --depth=4"
command = f'python Main.py {args} --log --agent --test' + option
print(command)
os.system(command)
