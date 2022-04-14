import os
import sys
args = ' '.join(map(str, sys.argv[1:]))
option = " --dataset=MNIST --in_dim=784 --hid_dim=256 --out_dim=256 --depth=6 --label_augmentation"
command = f'python Main.py {args} --log --agent --test' + option
print(command)
os.system(command)
