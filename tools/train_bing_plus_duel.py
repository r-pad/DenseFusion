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
from lib.network import PoseNetPlusDuelBing
from lib.loss import Loss
from lib.loss_bing import DuelLoss
from lib.utils import setup_logger
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes
from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.image_processing import ColorJitter, ImageNormalizer
from object_pose_utils.datasets.point_processing import PointShifter
from object_pose_utils.datasets.inplane_rotation_augmentation import InplaneRotator
from object_pose_utils.datasets.ycb_occlusion_augmentation import YCBOcclusionAugmentor
from object_pose_utils.utils.pose_processing import tensorAngularDiff
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--finetune_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.num_objects = 21 #number of object classes in the dataset
    opt.num_points = 1000 #number of points on the input pointcloud
    opt.outf = 'trained_models/ycb_plus_bing' #folder to save trained models
    opt.log_dir = 'experiments/logs/ycb_plus_bing' #folder to save logs
    opt.repeat_epoch = 1 #number of repeat times for one epoch training
    estimator = PoseNetPlusDuelBing(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()

    train_writer = SummaryWriter(comment='duel_binham_train')
    valid_writer = SummaryWriter(comment='duel_binham_valid')

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
    elif opt.finetune_posenet != '':
        pretrained_dict = torch.load(opt.finetune_posenet)
        model_dict = estimator.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        estimator.load_state_dict(model_dict)
        for k, v in estimator.named_parameters():
            if(k in pretrained_dict):
                v.requires_grad = False
        opt.log_dir += '_cont'
        opt.outf += '_cont'

    opt.refine_start = False
    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    object_list = list(range(1,22))
    output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                     otypes.IMAGE_CROPPED,
                     otypes.QUATERNION,
                     otypes.MODEL_POINTS_TRANSFORMED,
                     otypes.MODEL_POINTS,
                     otypes.OBJECT_LABEL,
                     ]

    dataset = YCBDataset(opt.dataset_root, mode='train_syn_grid_valid', 
                         object_list = object_list, 
                         output_data = output_format,
                         resample_on_error = True,
                         preprocessors = [YCBOcclusionAugmentor(opt.dataset_root), 
                                          ColorJitter(), 
                                          #InplaneRotator(),
                                          ],
                         postprocessors = [ImageNormalizer(), PointShifter()],
                         image_size = [640, 480], num_points=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = True, num_workers=opt.workers-1)
    
    test_dataset = YCBDataset(opt.dataset_root, mode='valid', 
                         object_list = object_list, 
                         output_data = output_format,
                         resample_on_error = True,
                         preprocessors = [],
                         postprocessors = [ImageNormalizer()],
                         image_size = [640, 480], num_points=1000)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = False, num_workers=1)
    

    opt.sym_list = [12, 15, 18, 19, 20]
    opt.num_points_mesh = dataset.num_pt_mesh_small

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion_dist = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_lik = DuelLoss(opt.num_points_mesh, opt.sym_list)

    best_dis = np.Inf
    best_lik = -np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()
    cum_batch_count = 0
    mean_err = 0
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        train_lik_avg = 0.0
        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, quat, target, model_points, idx = data
                idx = idx - 1
                points, choose, img, quat, target, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(quat).cuda(), \
                                                                 Variable(target).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
                pred_r, pred_t, pred_c, pred_bq, pred_bz, emb = estimator(img, points, choose, idx)
                loss_dist, dis, new_points, new_target = criterion_dist(pred_r, pred_t, pred_c, target, 
                        model_points, idx, points, opt.w, opt.refine_start)

                how_max, which_max = torch.max(pred_c.detach(), 1)
                pred_q = pred_r[0,:,[1,2,3,0]].detach()
                pred_q /= torch.norm(pred_q, dim=1).view(-1,1)

                max_q = pred_q[which_max.item()]
                max_bq = pred_bq[0,which_max.item()]/torch.norm(pred_bq[0,which_max.item()])
                max_bz = pred_bz[0,which_max.item()]

                loss_lik, lik = criterion_lik(max_q.view(-1), 
                    max_bq.view(-1), -torch.abs(max_bz.view(-1)), quat)
                loss = loss_dist + loss_lik
                loss.backward()

                train_dis_avg += dis.item()
                train_lik_avg += np.log(lik.item())
                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4} Avg_lik:{5}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size, train_lik_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0
                    train_lik_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_lik = 0.0
        test_count = 0
        estimator.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, quat, target, model_points, idx = data
            idx = idx - 1
            points, choose, img, quat, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(quat).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, pred_bq, pred_bz, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion_dist(pred_r, pred_t, pred_c, target, 
                    model_points, idx, points, opt.w, opt.refine_start)
            how_max, which_max = torch.max(pred_c.detach(), 1)
            pred_q = pred_r[0,:,[1,2,3,0]].detach()
            pred_q /= torch.norm(pred_q, dim=1).view(-1,1)

            max_q = pred_q[which_max.item()]
            max_bq = pred_bq[0,which_max.item()]/torch.norm(pred_bq[0,which_max.item()])
            max_bz = pred_bz[0,which_max.item()]

            _, lik = criterion_lik(max_q.view(-1), 
                max_bq.view(-1), -torch.abs(max_bz.view(-1)), quat)
            
            test_dis += dis.item()
            test_lik += np.log(lik.item())
            logger.info('Test time {0} Test Frame No.{1} dis:{2} lik:{3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis, lik))

            test_count += 1

        test_dis = test_dis / test_count
        test_lik = test_lik / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2} Avg lik: {3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis, test_lik))
        if test_dis <= best_dis or test_lik >= best_lik:
            best_dis = min(test_dis, best_dis)
            best_lik = max(test_lik, best_lik)

            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}_{3}.pth'.format(opt.outf, epoch, test_dis, test_lik))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_dis < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)


if __name__ == '__main__':
    main()
