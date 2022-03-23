#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/o.$JOB_ID
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
if [ "${1}" = "tanh-eye" ];then
  wandb agent tatsukichishibuya/InvTP/a790bmal
elif [ "${1}" = "tanh-uniform-1" ];then
  wandb agent tatsukichishibuya/InvTP/a2yn610o
elif [ "${1}" = "tanh-uniform-3" ];then
  wandb agent tatsukichishibuya/InvTP/z16z0pvm
elif [ "${1}" = "tanh-uniform-5" ];then
  wandb agent tatsukichishibuya/InvTP/cjxwi37o
elif [ "${1}" = "tanh-gaussian-1" ];then
  wandb agent tatsukichishibuya/InvTP/04bgwf5m
elif [ "${1}" = "tanh-gaussian-3" ];then
  wandb agent tatsukichishibuya/InvTP/gfxaixhk
elif [ "${1}" = "tanh-gaussian-5" ];then
  wandb agent tatsukichishibuya/InvTP/xm3w0xkz
elif [ "${1}" = "linear-eye" ];then
  wandb agent tatsukichishibuya/InvTP/r4i33k82
elif [ "${1}" = "linear-uniform-1" ];then
  wandb agent tatsukichishibuya/InvTP/7y7582am
elif [ "${1}" = "linear-uniform-3" ];then
  wandb agent tatsukichishibuya/InvTP/ohdjnvne
elif [ "${1}" = "linear-uniform-5" ];then
  wandb agent tatsukichishibuya/InvTP/wzil3sn2
elif [ "${1}" = "linear-gaussian-1" ];then
  wandb agent tatsukichishibuya/InvTP/yfzed0n9
elif [ "${1}" = "linear-gaussian-3" ];then
  wandb agent tatsukichishibuya/InvTP/5bqhimz4
elif [ "${1}" = "linear-gaussian-5" ];then
  wandb agent tatsukichishibuya/InvTP/55ceof6b
fi
