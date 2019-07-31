from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
#from dense_fusion.bingham_const import bingham_const
#from object_pose_utils.utils.interpolation import binghamNormC
from object_pose_utils.utils import to_np
from object_pose_utils.utils.bingham import iso_loss_calculation, duel_loss_calculation

class IsoLoss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(IsoLoss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_mean, pred_sigma, true_r):
        loss, lik = iso_loss_calculation(pred_mean, pred_sigma, true_r)
        return loss.mean(), lik.mean()


class DuelLoss(_Loss):
    def __init__(self, num_points_mesh, sym_list):
        super(DuelLoss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_q1 ,pred_q2, pred_z, true_r):
        loss, lik = duel_loss_calculation(pred_q1 ,pred_q2, pred_z, true_r)
        return loss.mean(), lik.mean()

