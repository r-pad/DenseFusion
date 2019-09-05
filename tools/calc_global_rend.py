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
from object_pose_utils.utils import to_np
from object_pose_utils.datasets.pose_dataset import processImage, processDepthImage
from object_pose_utils.datasets.image_processing import get_bbox_label, norm

from lib.network import PoseNetGlobal
from lib.utils import setup_logger
from tqdm import tqdm, trange

from quat_math import euler_matrix, quaternion_matrix, quaternion_about_axis

def getYCBTransform(q, t=[0,0,1]):
    trans_mat = quaternion_matrix(q)
    ycb_mat = euler_matrix(-np.pi/2,0,0)
    trans_mat = trans_mat.dot(ycb_mat)
    trans_mat[:3,3] = t
    return trans_mat

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
    output_format = [otypes.QUATERNION,]
    estimator.eval()

    fx = 1066.778
    fy = 1067.487
    px = 312.9869
    py = 241.3109
    filename_template = dataset_root + '/data/{}-{}-{}.png'

    for cls in [1]:#trange(1,num_objects+1):
        dataset = YCBDataset(opt.dataset_root, mode = opt.mode, object_list = [cls], 
                             output_data = output_format,
                             image_size = [640, 480], num_points = 1000)
        classes = dataset.classes
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

        for i, q in tqdm(enumerate(dataloader), total = len(dataloader)):
            if(len(data) == 0 or len(data[0]) == 0):
                continue
            subpath = dataset.getPath(i)
            trans_mat = getYCBTransform(q[0], [0,0,1])
            img = np.array(Image.open(filename_template.format(subpath, cls, 'render')))
            depth = np.array(Image.open(filename_template.format(subpath, cls, 'rdepth')))

            meta_data_rend = {}
            meta_data_rend['transform_mat'] = trans_mat
            meta_data_rend['camera_scale'] = 10000
            meta_data_rend['camera_fx'] = fx
            meta_data_rend['camera_fy'] = fy
            meta_data_rend['camera_cx'] = px
            meta_data_rend['camera_cy'] = py
            meta_data_rend['mask'] = img[:,:,3] > 128
            meta_data_rend['bbox'] = get_bbox_label(meta_data_rend['mask'])

            img = norm(processImage(img[:,:,:3], meta_data_rend, otypes.IMAGE_MASKED_CROPPED))
            points, choose = processDepthImage(depth, meta_data_rend, otypes.DEPTH_POINTS_MASKED_AND_INDEXES)
            
            img = Variable(img).cuda()
            points = Variable(points).cuda()
            choose = Variable(choose).cuda()
            idx = Variable(torch.LongTensor(cls-1)).cuda()
            #assert cls == data_path[1]
            feat, _ = estimator.globalFeature(img, points, choose, idx)
            output_filename = '{0}/{1}_{2}_rfeat.npy'.format(opt.output_folder, subpath, classes[cls])
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            np.save(output_filename, to_np(feat)[0])
            
       
if __name__ == '__main__':
    main()
