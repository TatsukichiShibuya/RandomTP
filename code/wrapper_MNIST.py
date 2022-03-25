import os
import sys
args = ' '.join(map(str, sys.argv[1:]))
mnist = " --dataset=MNIST --in_dim=784 --hid_dim=256 --out_dim=256 --label_augmentation "
command = f'python Main.py {args} --log --agent' + mnist
print(command)
os.system(command)
