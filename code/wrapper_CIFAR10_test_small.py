import os
import sys
args = ' '.join(map(str, sys.argv[1:]))
option = " --dataset=CIFAR10 --in_dim=3072 --hid_dim=632 --out_dim=10 --depth=4"
command = f'python Main.py {args} --log --agent --test' + option
print(command)
os.system(command)
