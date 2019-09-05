import glob
import numpy as np
from PIL import Image
import cv2 
import quat_math as tfs
import scipy.io as scio
from tqdm import tqdm

def main():
    dataset_root = 'datasets/ycb/YCB_Video_Dataset'
    
    with open('datasets/ycb/dataset_config/classes.txt') as f: 
        classes = f.read().split()
    classes.insert(0,'background')

#    with open('datasets/ycb/dataset_config/rendered_data_list.txt') as f: 
#        file_list = f.read().split()

    for obj in tqdm(range(2, 22)):
        file_list = ['depth_renders_offset/{}/{:04d}'.format(classes[obj], i) for i in range(3885)]

        for fn in tqdm(file_list):
            img = Image.open('{0}/{1}-color.png'.format(dataset_root, fn))
            obj_label = classes.index(fn.split('/')[-2]) 
            quat = np.load('{0}/{1}-trans.npy'.format(dataset_root, fn))
            label = np.where(np.array(img.split()[-1])==255, obj_label, 0)
            cv2.imwrite('{0}/{1}-label.png'.format(dataset_root, fn), label)	
            poses = np.zeros([3,4,1])
            poses[:3,:3,0] = tfs.quaternion_matrix(quat)[:3,:3]
            poses[:3,3,0] = [0.,0.,1.]
            scio.savemat('{0}/{1}-meta.mat'.format(dataset_root, fn), 
                         {'cls_indexes':np.array([[obj_label]]), 
                          'factor_depth':np.array([[10000]]),
                          'poses':poses})

if __name__=='__main__':
    main()
