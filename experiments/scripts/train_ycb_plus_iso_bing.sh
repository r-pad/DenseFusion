#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train_bing_plus_iso.py \
  --dataset_root ./datasets/ycb/YCB_Video_Dataset \
  --finetune_posenet ./trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth
#  --resume_posenet pose_model_current.pth
