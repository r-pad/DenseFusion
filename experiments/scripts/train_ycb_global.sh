#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 ./tools/train_with_global_feature.py --dataset ycb\
  --dataset_root datasets/ycb/YCB_Video_Dataset
