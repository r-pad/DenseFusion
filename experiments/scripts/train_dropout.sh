#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train_dropout.py --dataset ycb\
  --dataset_root ./datasets/ycb/YCB_Video_Dataset
