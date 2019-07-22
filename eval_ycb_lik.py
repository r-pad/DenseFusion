

import cv2
from tqdm import tqdm
import numpy as np
import scipy.io as scio

import torch
from PIL import Image
from functools import partial

import quat_math as qm
from generic_pose.utils import to_np, to_var
from dense_fusion.evaluate import DenseFusionEstimator
from dense_fusion.evaluate_likelihood import evaluateYCBDataset
from dense_fusion.evaluate_likelihood import subRandomSigmaSearchMax, subRandomSigmaSearchEvery


num_points = 1000
num_obj = 21

dataset_root = './datasets/ycb/YCB_Video_Dataset'
#model_checkpoint = 'trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth'
#refine_model_checkpoint = 'trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth'
valid_model_checkpoint = '../DenseFusionOld/DenseFusion/trained_models/ycb/pose_model_34_0.025648579025031315.pth'

dataset_config_dir = 'datasets/ycb/dataset_config'
#dataset_config_dir = '../DenseFusionOld/DenseFusion/datasets/ycb/dataset_config'
test_filenames = '{0}/test_data_list.txt'.format(dataset_config_dir)

df_estimator = DenseFusionEstimator(num_points, num_obj, valid_model_checkpoint)

subRandomSigmaSearchMax(df_estimator, dataset_root, test_filenames, 
                     sigma_lims = [0, 100],
                     num_samples=20)

subRandomSigmaSearchEvery(df_estimator, dataset_root, test_filenames, 
                     sigma_lims = [0, 100],
                     num_samples=20)


#data = np.load('single_max.npz')
#likelihoods = data['likelihoods']
#sigma = data['sigma']
#print('Using sigma {}'.format(sigma))
#eval_func = partial(evaluateYCBMax, df_estimator, sigma=sigma)
#likelihoods = evaluateYCBDataset(eval_func, dataset_root, test_filenames)
#np.savez('single_max_{}.npz'.format(sigma), likelihoods = likelihoods, sigma = sigma)
