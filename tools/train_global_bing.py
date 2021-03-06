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
from lib.network import PoseNetBingham
from lib.loss_bing import IsoLoss
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
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()


def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    opt.num_objects = 21 #number of object classes in the dataset
    opt.num_points = 1000 #number of points on the input pointcloud
    opt.outf = 'trained_models/ycb_global_bing' #folder to save trained models
    opt.log_dir = 'experiments/logs/ycb_global_bing' #folder to save logs
    opt.repeat_epoch = 1 #number of repeat times for one epoch training
    estimator = PoseNetBingham(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()

    train_writer = SummaryWriter(comment='binham_train')
    valid_writer = SummaryWriter(comment='binham_valid')


    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    object_list = list(range(1,22))
    output_format = [otypes.DEPTH_POINTS_MASKED_AND_INDEXES,
                     otypes.IMAGE_CROPPED,
                     otypes.QUATERNION,
                     otypes.OBJECT_LABEL,
                     ]

    dataset = YCBDataset(opt.dataset_root, mode='train_syn_grid_valid', 
                         object_list = object_list, 
                         output_data = output_format,
                         resample_on_error = True,
                         preprocessors = [YCBOcclusionAugmentor(opt.dataset_root), 
                                          ColorJitter(), 
                                          InplaneRotator()],
                         postprocessors = [ImageNormalizer(), PointShifter()],
                         image_size = [640, 480], num_points=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers-1)
    
    test_dataset = YCBDataset(opt.dataset_root, mode='valid', 
                         object_list = object_list, 
                         output_data = output_format,
                         resample_on_error = True,
                         preprocessors = [],
                         postprocessors = [ImageNormalizer()],
                         image_size = [640, 480], num_points=1000)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    

    opt.sym_list = [12, 15, 18, 19, 20]
    opt.num_points_mesh = dataset.num_pt_mesh_small

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = IsoLoss(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()
    cum_batch_count = 0
    mean_sig = 0
    mean_err = 0
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, idx = data
                idx = idx - 1
                points, choose, img, target, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target).cuda(), \
                                                                 Variable(idx).cuda()
                pred_mean, pred_sigma, _ = estimator(img, points, choose, idx)
                loss, dis = criterion(pred_mean.view(-1), torch.abs(pred_sigma[:,:,0].view(-1)), target)
                mean_sig += torch.abs(pred_sigma[:,:,0].view(-1)).detach()
                pred_q = pred_mean.view(-1).detach()
                pred_q = pred_q /pred_q.norm()
                mean_err += tensorAngularDiff(pred_q, target)*180/np.pi
                loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    cum_batch_count += 1
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count/len(dataset), train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    if(cum_batch_count % 100 == 0):
                        train_writer.add_scalar('loss', loss, cum_batch_count)
                        train_writer.add_scalar('lik', dis, cum_batch_count)
                        train_writer.add_scalar('mean_lik', train_dis_avg / opt.batch_size, cum_batch_count)
                        train_writer.add_scalar('mean_sig', mean_sig/(100*opt.batch_size), cum_batch_count)
                        train_writer.add_scalar('mean_err', mean_err/(100*opt.batch_size), cum_batch_count)
                        mean_sig = 0
                        mean_err = 0
                    train_dis_avg = 0
                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))
                
                if(train_count >= 100000):
                    break

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        mean_sig = 0
        mean_err = 0
        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, idx = data
            idx = idx - 1
            points, choose, img, target, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(idx).cuda()
            pred_mean, pred_sigma, _ = estimator(img, points, choose, idx)
            loss, dis = criterion(pred_mean.view(-1), torch.abs(pred_sigma[:,:,0].view(-1)), target)
            mean_sig += torch.abs(pred_sigma[:,:,0].view(-1)).detach()
            pred_q = pred_mean.view(-1).detach()
            pred_q = pred_q /pred_q.norm()
            mean_err += tensorAngularDiff(pred_q, target)*180/np.pi

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
        test_count += 1
        if(test_count >= 3000):
            break

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        valid_writer.add_scalar('loss', loss, cum_batch_count)
        valid_writer.add_scalar('lik', dis, cum_batch_count)
        valid_writer.add_scalar('mean_lik', test_dis, cum_batch_count)
        valid_writer.add_scalar('mean_sig', mean_sig/test_count, cum_batch_count)
        valid_writer.add_scalar('mean_err', mean_err/test_count, cum_batch_count)
        mean_sig = 0
        mean_err = 0
        if test_dis <= best_test:
            best_test = test_dis
            torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

if __name__ == '__main__':
    main()
