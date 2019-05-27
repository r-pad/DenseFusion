# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.ycb_dataset import YcbImagePreprocessor
from object_pose_utils.utils import to_np

from lib.network import PoseNetGlobal
from lib.utils import setup_logger
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default =  'datasets/ycb/YCB_Video_Dataset', 
        help='dataset root dir (''YCB_Video_Dataset'')')
parser.add_argument('--mode', type=str, default = 'train_syn_grid', help='Dataset mode')
parser.add_argument('--batch_size', type=int, default = 1, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--weights', type=str, help='PoseNetGlobal weights file')
parser.add_argument('--output_folder', type=str, help='Feature save location')
opt = parser.parse_args()


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    num_points = 1000 #number of points on the input pointcloud
    num_objects = 21 
    estimator = PoseNetGlobal(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.weights))
    output_format = [otypes.QUATERNION, otypes.IMAGE_CROPPED, 
                     otypes.DEPTH_POINTS_MASKED_AND_INDEXES]
    estimator.eval()
    for cls in trange(1,num_objects+1):
        dataset = YCBDataset(opt.dataset_root, mode = opt.mode, object_list = [cls], 
                             output_data = output_format,
                             preprocessor = YcbImagePreprocessor,
                             image_size = [640, 480], num_points = 1000)
        classes = dataset.classes
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

        for i, data in tqdm(enumerate(dataloader), total = len(dataloader)):
            if(len(data) == 0 or len(data[0]) == 0):
                continue
            quat, img, points, choose = data
            data_path = dataset.image_list[i]
            img = Variable(img).cuda()
            points = Variable(points).cuda()
            choose = Variable(choose).cuda()
            idx = Variable(torch.LongTensor(cls-1)).cuda()
            assert cls == data_path[1]
            feat, _ = estimator.globalFeature(img, points, choose, idx)
            output_filename = '{0}/{1}_{2}_feat.npz'.format(opt.output_folder, data_path[0], classes[cls])
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            np.savez(output_filename, quat = to_np(quat)[0], feat = to_np(feat)[0])
            
       
if __name__ == '__main__':
    main()
