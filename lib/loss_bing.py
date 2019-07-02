from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
from dense_fusion.bingham_const import bingham_const
#from object_pose_utils.utils.interpolation import binghamNormC
from object_pose_utils.utils import to_np


def makeDuelMatrix(q):
    a, b, c, d = q 
    mat = torch.stack([torch.stack([a, -b, -c, -d]),
                       torch.stack([b,  a, -d,  c]), 
                       torch.stack([c,  d,  a, -b]),
                       torch.stack([d, -c,  b,  a])])
    return mat

def makeDuelMatrix2(q2):
    p, q, r, s = q2
    mat = torch.stack([torch.stack([p, -q, -r, -s]),
                       torch.stack([q,  p,  s, -r]), 
                       torch.stack([r, -s,  p,  q]),
                       torch.stack([s,  r, -q,  p])])
    return mat 

def makeBinghamM(q, q2 = None):
    q = q/q.norm()
    if q2 is None:
        return makeDuelMatrix(q)
    q2 = q2/q2.norm()
    return torch.mm(makeDuelMatrix(q),
                    makeDuelMatrix2(q2))

def bingham_likelihood(M, Z, label, return_exponent = False):
    Z = Z.clamp(max=0, min=-1000)
    eta = bingham_const(Z[1:]).float()
    if(torch.cuda.is_available()):
        eta = eta.cuda()

    Z = torch.diag(Z)
    MZMt = torch.bmm(torch.bmm(M, Z.repeat([1,1,1])), torch.transpose(M,2,1))
    if(torch.cuda.is_available()):
        MZMt = MZMt.cuda()
    bingham_p = torch.mul(label.transpose(1,0).unsqueeze(2),
    torch.matmul(label,MZMt.transpose(2,0))).sum([0])
    if(return_exponent):
        return bingham_p, eta 
    else:
        bingham_p = 1./eta*torch.exp(bingham_p)
    return bingham_p

def isobingham_likelihood(mean, sigma, label, return_exponent = False):
    M = makeBinghamM(mean).unsqueeze(0)
    zero = torch.zeros(1).float()
    if(torch.cuda.is_available()):
        zero = zero.cuda()
    Z = torch.cat([zero,-sigma, -sigma, -sigma])
    return bingham_likelihood(M, Z, label, return_exponent)

def duel_quat_bingham_likelihood(q1, q2, z, label, return_exponent = False):
    M = makeBinghamM(q1, q2).unsqueeze(0)
    zero = torch.zeros(1).float()
    if(torch.cuda.is_available()):
        zero = zero.cuda()
    Z = torch.cat([zero, z])
    return bingham_likelihood(M, Z, label, return_exponent)

def duel_loss_calculation(pred_q1, pred_q2, pred_z, true_r):
    lik_exp, eta = duel_quat_bingham_likelihood(pred_q1, pred_q2, pred_z, true_r, return_exponent = True)
    lik = 1./eta*torch.exp(lik_exp)
    loss = -(lik_exp - torch.log(eta))
    #if(lik != lik):
    #    raise ValueError('NAN lik: {} for mean {} and sigma {}'.format(lik, pred_mean, pred_sigma))
    return loss, lik 

def iso_loss_calculation(pred_mean, pred_sigma, true_r):
    lik_exp, eta = isobingham_likelihood(pred_mean, pred_sigma, true_r, return_exponent = True)
    lik = 1./eta*torch.exp(lik_exp)
    loss = -(lik_exp - torch.log(eta))
    #if(lik != lik):
    #    raise ValueError('NAN lik: {} for mean {} and sigma {}'.format(lik, pred_mean, pred_sigma))
    return loss, lik 

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

    def forward(self, pred_mean, pred_sigma, true_r):
        loss, lik = duel_loss_calculation(pred_q1 ,pred_q2, pred_z, true_r)
        return loss.mean(), lik.mean()

