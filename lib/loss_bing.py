from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
from object_pose_utils.utils.interpolation import binghamNormC
from object_pose_utils.utils import to_np

def proj(u, v):
    return v.dot(u)/u.dot(u)*u

def gramSchmidt(v):
    n = 4
    B = []
    for j in range(n):
        u = v.squeeze()
        for k in range(j):
            u = u - proj(B[k].squeeze(), v)
        B.append(u/torch.norm(u))
        v = torch.randn(n).cuda()
    return torch.stack(B)

def bingham_likelihood(mean, sigma, label, return_exponent = False):
    sigma = sigma.clamp(0,100)
    M = gramSchmidt(mean).unsqueeze(0).cuda()
    Z = torch.diag(torch.Tensor([0,-sigma, -sigma, -sigma])).cuda()
    MZMt = torch.bmm(torch.bmm(M, Z.repeat([1,1,1])), torch.transpose(M,2,1))
    if(torch.cuda.is_available()):
        MZMt = MZMt.cuda()
    eta = binghamNormC(to_np(sigma).flat[0])
    bingham_p = torch.mul(label.transpose(1,0).unsqueeze(2),
    torch.matmul(label,MZMt.transpose(2,0))).sum([0])
   
    if(return_exponent):
        return bingham_p, eta
    else:
        bingham_p = 1./eta*torch.exp(bingham_p)
    return bingham_p

def loss_calculation(pred_mean, pred_sigma, true_r, sym_list):
    lik_exp, eta = bingham_likelihood(pred_mean, pred_sigma, true_r, return_exponent = True)
    lik = 1./eta*torch.exp(lik_exp)
    loss = -(lik - np.log(eta))
    if(lik != lik):
        raise ValueError('NAN lik: {} for mean {} and sigma {}'.format(lik, pred_mean, pred_sigma))
    return loss, lik

class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_mean, pred_sigma, true_r):
        return loss_calculation(pred_mean, pred_sigma, true_r, self.sym_list)

