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
from object_pose_utils.datasets.image_processing import ColorJitter, ImageNormalizer
from object_pose_utils.datasets.ycb_occlusion_augmentation import YCBOcclusionAugmentor
from object_pose_utils.datasets.point_processing import PointShifter
from object_pose_utils.utils import to_np

from lib.network import PoseNet
from lib.utils import setup_logger
from tqdm import tqdm, trange


from time import sleep
import contextlib
import sys

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default =  'datasets/ycb/YCB_Video_Dataset', 
        help='Dataset root dir (''YCB_Video_Dataset'')')
parser.add_argument('--num_augmentations', type=int, default = 0, 
        help='Number of augmented images per render')
parser.add_argument('--dataset_mode', type=str, default = 'train_syn_valid',
        help='Dataset mode')
parser.add_argument('--no_background', dest='add_syn_background', action='store_false')
parser.add_argument('--workers', type=int, default = 10, help='Number of data loading workers')
parser.add_argument('--weights', type=str, help='PoseNetGlobal weights file')
parser.add_argument('--output_folder', type=str, help='Feature save location')
parser.add_argument('--object_indices', type=int, nargs='+', default = None, help='Object indices to featureize')
parser.add_argument('--start_index', type=int, default = 0, help='Starting augmentation index')
opt = parser.parse_args()


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    num_points = 1000 #number of points on the input pointcloud
    num_objects = 21 
    if(opt.object_indices is None):
        opt.object_indices = list(range(1,num_objects+1))
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.weights))
    output_format = [otypes.OBJECT_LABEL,
                     otypes.QUATERNION, otypes.IMAGE_CROPPED, 
                     otypes.DEPTH_POINTS_MASKED_AND_INDEXES]
    estimator.eval()

    with std_out_err_redirect_tqdm() as orig_stdout:
        pbar = tqdm(opt.object_indices, file=orig_stdout, dynamic_ncols=True) 
        for cls in pbar:
            preprocessors = []
            postprocessors = [ImageNormalizer()]
            if(opt.num_augmentations > 0):
                preprocessors.extend([YCBOcclusionAugmentor(opt.dataset_root), 
                                      ColorJitter(),])
                postprocessors.append(PointShifter())
            
            dataset = YCBDataset(opt.dataset_root, mode=opt.dataset_mode,
                                 object_list = [cls], 
                                 output_data = output_format,
                                 resample_on_error = False,
                                 add_syn_background = opt.add_syn_background,
                                 add_syn_noise = opt.add_syn_background,
                                 preprocessors = preprocessors, 
                                 postprocessors = postprocessors,
                                 image_size = [640, 480], num_points=1000)
         
            classes = dataset.classes
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            pbar.set_description('Featurizing {}'.format(classes[cls]))
            if(opt.num_augmentations > 0):
                pbar_aug = trange(opt.start_index, opt.num_augmentations, file=orig_stdout, dynamic_ncols=True)
            else:
                pbar_aug = [None]
            for aug_idx in pbar_aug:
                pbar_save = tqdm(enumerate(dataloader), total = len(dataloader),
                                 file=orig_stdout, dynamic_ncols=True)
                for i, data in pbar_save:
                    if(len(data) == 0 or len(data[0]) == 0):
                        continue
                    idx, quat, img, points, choose = data
                    data_path = dataset.image_list[i]
                    idx = idx - 1
                    img = Variable(img).cuda()
                    points = Variable(points).cuda()
                    choose = Variable(choose).cuda()
                    idx = Variable(idx).cuda()
                    assert cls == data_path[1]
                    assert cls - 1 == int(idx[0])
                    pred_r, pred_t, pred_c, emb, feat, feat_global = estimator.allFeatures(img, points, choose, idx)
                    if(opt.num_augmentations > 0):
                        output_filename = '{0}/data/{1}_{2}_{3}_feat.npz'.format(opt.output_folder, 
                                data_path[0], classes[cls], aug_idx)
                    else:
                        output_filename = '{0}/data/{1}_{2}_feat.npz'.format(opt.output_folder, 
                                data_path[0], classes[cls])
                    #pbar_save.set_description(output_filename)
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    how_max, which_max = torch.max(pred_c, 1)
                    max_feat = feat[0,:,which_max[0]].view(-1)

                    np.savez(output_filename, 
                             quat = to_np(quat)[0], 
                             feat = to_np(max_feat),
                             #feat_all = to_np(feat)[0].T, i
                             feat_global = to_np(feat_global)[0], 
                             #max_confidence = to_np(how_max),
                             #confidence = to_np(pred_c)[0],
                             )
                    
               
if __name__ == '__main__':
    main()
