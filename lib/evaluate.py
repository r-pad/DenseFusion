import cv2
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import numpy as np
import quat_math as qm
from generic_pose.utils import to_np, to_var

import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

def get_bbox(posecnn_rois, idx = 0, img_width=480, img_length=640):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def preprocessData(img, depth, posecnn_meta, idx = 0, 
                   num_points = 1000, img_width = 480, img_length = 640, 
                   cam_cx = 312.9869, cam_cy = 241.3109,
                   cam_fx = 1066.778, cam_fy = 1067.487,
                   cam_scale = 10000.0):
    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])
    lst = posecnn_rois[:, 1:2].flatten()
    if(idx > len(lst)):
        return None
    itemid = lst[idx]

    rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, idx, img_width, img_length)

    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
    mask = mask_label * mask_depth

    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    elif len(choose) > 0:
        choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
    else:
        return None

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    img_masked = np.array(img)[:, :, :3]
    img_masked = np.transpose(img_masked, (2, 0, 1))
    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

    cloud = torch.from_numpy(cloud.astype(np.float32))
    choose = torch.LongTensor(choose.astype(np.int32))
    img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
    index = torch.LongTensor([itemid - 1])

    cloud = Variable(cloud).cuda()
    choose = Variable(choose).cuda()
    img_masked = Variable(img_masked).cuda()
    index = Variable(index).cuda()

    cloud = cloud.view(1, num_points, 3)
    img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

    return img_masked, cloud, choose, index

def getCoords(choose, masked_shape):
    return np.unravel_index(to_np(choose), masked_shape[2:])

def getGroundtruth(posecnn_meta, obj_idx):
    posecnn_idx = posecnn_meta['rois'][obj_idx, 1]
    pose_idx = np.where(pose_meta['cls_indexes'].flatten()==posecnn_idx)[0][0]
    mat_gt = pose_meta['poses'][:,:, pose_idx]
    R_gt = np.eye(4)
    R_gt[:3,:3] = mat_gt[:3,:3]
    q_gt = qm.quaternion_from_matrix(R_gt)
    t_gt = mat_gt[:3,3]
    return q_gt, t_gt

class DenseFusionEstimator(object):    
    def __init__(self, num_points, num_obj, estimator_weights_file, 
                 refiner_weights_file = None,
                 batch_size = 1):
        self.num_points = num_points
        self.bs = batch_size
        self.estimator = PoseNet(num_points = num_points, num_obj = num_obj)
        if(torch.cuda.is_available()):
            self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(estimator_weights_file))
        self.estimator.eval()

        if(refiner_weights_file is not None):
            self.refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
            if(torch.cuda.is_available()):
                self.refiner.cuda()
            self.refiner.load_state_dict(torch.load(refiner_weights_file))
            self.refiner.eval()
        else:
            self.refiner = None
    
    def __call__(self, img, depth, posecnn_meta, idx = 0):
        data = preprocessData(img, depth, posecnn_meta, idx,
                              num_points = self.num_points)
        if(data is None):
            return None, None
        img_masked, cloud, choose, model_index = data
        max_r, max_t, max_c, pred_r, pred_t, pred_c, emb = self.estimatePose(img_masked, cloud, choose, 
                model_index, return_all = True)
        if(self.refiner is not None):
            refined_r, refined_t = self.refinePose(emb, cloud, model_index, max_r, max_t)
            return refined_r, refined_t

        return max_r, max_t

    def globalFeature(self, img_masked, cloud, choose, index):
        _, _, _, emb = self.estimator(img_masked, cloud, choose, index)
        return emb

    def estimatePose(self, img_masked, cloud, choose, index, return_all=True):
        pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)
        pred_c = pred_c.view(self.bs, self.num_points)
        max_c, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
        points = cloud.view(self.bs * self.num_points, 1, 3)
        
        max_r = pred_r[0][which_max[0]].view(-1)#.cpu().data.numpy()
        max_t = (points + pred_t)[which_max[0]].view(-1)#.cpu().data.numpy()
        #max_c = max_c.cpu().data.numpy()
        
        if(return_all):
            return max_r, max_t, max_c, pred_r, pred_t, pred_c, emb
        
        return max_r, max_t, max_c

    def refinePose(self, emb, cloud, index, init_t, init_r, iterations = 2):
        init_t = init_t.cpu().data.numpy()
        init_r = init_r.cpu().data.numpy()

        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(init_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, self.num_points, 3)
            init_mat = quaternion_matrix(init_r)
            R = Variable(torch.from_numpy(init_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            init_mat[0:3, 3] = init_t

            new_cloud = torch.bmm((cloud - T), R).contiguous()
            pred_r, pred_t = self.refiner(new_cloud, emb, index)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            
            delta_r = pred_r.view(-1).cpu().data.numpy()
            delta_t = pred_t.view(-1).cpu().data.numpy()
            delta_mat = quaternion_matrix(delta_r)

            delta_mat[0:3, 3] = delta_t

            refined_mat = np.dot(init_mat, delta_mat)
            refined_r = copy.deepcopy(refined_mat)
            refined_r[0:3, 3] = 0
            refined_r = quaternion_from_matrix(refined_r, True)
            refined_t = np.array([refined_mat[0][3], refined_mat[1][3], refined_mat[2][3]])

            init_r = r_final
            init_t = t_final
        return refined_t, refined_t
      
