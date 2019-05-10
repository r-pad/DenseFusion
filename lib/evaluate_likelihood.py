import numpy as np
import scipy.io as scio
from PIL import Image

from tqdm import tqdm

import torch
from functools import partial

from object_pose_utils.utils import to_np
from object_pose_utils.utils.interpolation import BinghamInterpolation, binghamNormC
from object_pose_utils.utils.subrandom import subrandom
from dense_fusion.data_processing import preprocessPoseCNNMetaData, getYCBGroundtruth

def logSumExp(x, weights, eta):
    max_x = torch.max(x, dim=1)[0]
    l_x = max_x + torch.log((weights*torch.exp(x - max_x.unsqueeze(1))).sum(1)) - np.log(eta)
    return l_x

def evaluateYCBMax(estimator, data_prefix, sigma, return_exponent = False):
    img = Image.open('{}-color.png'.format(data_prefix))
    depth = np.array(Image.open('{}-depth.png'.format(data_prefix)))
    pose_meta = scio.loadmat('{}-meta.mat'.format(data_prefix))
    posecnn_meta = scio.loadmat('{}-posecnn.mat'.format(data_prefix))
    object_classes = set(pose_meta['cls_indexes'].flat) & \
                        set(posecnn_meta['rois'][:,1:2].flatten().astype(int))
    
    likelihood = {}
    for cls_idx in object_classes:
        obj_idx = np.nonzero(posecnn_meta['rois'][:,1].astype(int) == cls_idx)[0][0]
        mask, bbox, object_label = preprocessPoseCNNMetaData(posecnn_meta, obj_idx)
        q_est, t_est = estimator(img, depth, mask, bbox, object_label)
        q_est = q_est[[1,2,3,0]]
    
        bingham_interp = BinghamInterpolation(vertices = [to_np(q_est)], 
                                             values = torch.Tensor([1]), 
                                             sigma=sigma)
    
        q_gt, _ = getYCBGroundtruth(pose_meta, posecnn_meta, obj_idx)
        q_gt = torch.Tensor(q_gt).unsqueeze(0)
        if(torch.cuda.is_available()):
            q_gt = q_gt.cuda()
        likelihood[cls_idx] = (bingham_interp(q_gt, return_exponent), bingham_interp.values)
        #likelihood[cls_idx] = binghamInterp(q_gt)
    
    return likelihood

def evaluateYCBEvery(estimator, data_prefix, sigma, return_exponent = False):
    img = Image.open('{}-color.png'.format(data_prefix))
    depth = np.array(Image.open('{}-depth.png'.format(data_prefix)))
    pose_meta = scio.loadmat('{}-meta.mat'.format(data_prefix))
    posecnn_meta = scio.loadmat('{}-posecnn.mat'.format(data_prefix))
    object_classes = set(pose_meta['cls_indexes'].flat) & \
                        set(posecnn_meta['rois'][:,1:2].flatten().astype(int))
    
    likelihood = {}
    for cls_idx in object_classes:
        obj_idx = np.nonzero(posecnn_meta['rois'][:,1].astype(int) == cls_idx)[0][0]
        mask, bbox, object_label = preprocessPoseCNNMetaData(posecnn_meta, obj_idx)
        pred_r, _, pred_c = estimator(img, depth, mask, bbox, object_label, return_all = True)[3:6]
        pred_r = pred_r[0,:,[1,2,3,0]]
        bingham_interp = BinghamInterpolation(vertices = to_np(pred_r.detach()), 
                                              values = pred_c.detach(), 
                                              sigma = sigma)
    
        q_gt, _ = getYCBGroundtruth(pose_meta, posecnn_meta, obj_idx)
        q_gt = torch.Tensor(q_gt).unsqueeze(0)
        if(torch.cuda.is_available()):
            q_gt = q_gt.cuda()
        likelihood[cls_idx] = (bingham_interp(q_gt, return_exponent), bingham_interp.values)
    
    return likelihood


def getYCBClassData(dataset_root):
    with open('{0}/image_sets/classes.txt'.format(dataset_root)) as f:
        class_list = f.read().split()
    
    model_points = {}
    for cls_id, cls in enumerate(class_list[1:], 1):
        with open('{0}/models/{1}/points.xyz'.format(dataset_root, cls)) as f:
            model_points[cls_id] = np.loadtxt(f)
    
    return class_list, model_points 
 

def evaluateYCBDataset(eval_func, dataset_root, file_list, skip=0):
    with torch.no_grad():
        file_prefix = []
        if(type(file_list) is str):
            with open(file_list) as f:
                file_list = f.read().split()
        elif(type(file_list) not in (tuple, list)):
            raise TypeError('Invalid file_list type {}. Valid types: str, tuple, list'.format(type(file_list)))

        class_list, model_points = getYCBClassData(dataset_root)
        num_obj = len(class_list)-1

        likelihoods = {cls_idx:[] for cls_idx in range(1, num_obj+2)}
        file_list = file_list[::(skip+1)]
        pbar = tqdm(file_list)

        for data_prefix in pbar:
            pbar.set_description('Processing {}'.format(data_prefix))
            lh = eval_func('{0}/{1}'.format(dataset_root, data_prefix))
            for k,v in lh.items():
                likelihoods[k].append(v)

        return likelihoods

def subRandomSigmaSearchMax(estimator, dataset_root, file_list, 
                            sigma_lims = [0, 2*np.pi], 
                            num_samples = 100):
    max_likelihood = -np.inf
    max_sigma = None
    sigmas = subrandom(num_samples)*(sigma_lims[1]-sigma_lims[0]) + sigma_lims[0]
    mean_likelihoods = []
    for j, sigma in enumerate(sigmas):
        eval_func = partial(evaluateYCBMax, estimator, sigma=sigma, return_exponent=True)
        likelihoods = evaluateYCBDataset(eval_func, dataset_root, file_list, 20)
        mean_likelihood = 0
        n = 0
        for v in likelihoods.values():
            l_exp, w = zip(*v)
            l_exp = torch.cat(l_exp)
            w = torch.stack(w)
            mean_likelihood += to_np(torch.sum(logSumExp(l_exp, w, binghamNormC(sigma))))
            n += len(v)
        mean_likelihood /= n
        print("{}: Mean Log Likelihood of Sigma {}: {}".format(j, sigma, mean_likelihood))
        mean_likelihoods.append(mean_likelihood)
        if(mean_likelihood > max_likelihood):
            max_likelihood = mean_likelihood
            max_sigma = sigma
            print("Max Sigma after {} samples: {} ({})".format(j+1, sigma, max_likelihood))
            np.savez('single_max.npz', likelihoods = likelihoods, sigma = max_sigma)

    np.savez('single_max_sigmas.npz', mean_likelihoods = np.array(mean_likelihoods), sigmas = sigmas)

def subRandomSigmaSearchEvery(estimator, dataset_root, file_list, 
                              sigma_lims = [0, 100], 
                              num_samples = 100):
    max_likelihood = -np.inf
    max_sigma = None
    sigmas = subrandom(num_samples)*(sigma_lims[1]-sigma_lims[0]) + sigma_lims[0]
    mean_likelihoods = []
    for j, sigma in enumerate(sigmas):
        eval_func = partial(evaluateYCBEvery, estimator, sigma=sigma, return_exponent=True)
        likelihoods = evaluateYCBDataset(eval_func, dataset_root, file_list, 20)
        mean_likelihood = 0
        n = 0
        for v in likelihoods.values():
            l_exp, w = zip(*v)
            l_exp = torch.cat(l_exp)
            w = torch.stack(w).squeeze()
            mean_likelihood += to_np(torch.sum(logSumExp(l_exp, w, binghamNormC(sigma))))
            n += len(v)
        mean_likelihood /= n
        print("{}: Mean Log Likelihood of Sigma {}: {}".format(j, sigma, mean_likelihood))
        mean_likelihoods.append(mean_likelihood)
        if(mean_likelihood > max_likelihood):
            max_likelihood = mean_likelihood
            max_sigma = sigma
            print("Max Sigma after {} samples: {} ({})".format(j+1, sigma, max_likelihood))
            np.savez('bingham_every.npz', likelihoods = likelihoods, sigma = max_sigma)

    np.savez('bingham_every_sigmas.npz', mean_likelihoods = np.array(mean_likelihoods), sigmas = sigmas)

