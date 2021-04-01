import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


dataset_path = "D:/Projects/PanoLayout/PanoLayout_master/exp/full/output/depth_4"
gt_dataset_path = "D:/Projects/PanoLayout/data/"

#list_path = 'D:/Projects/PanoLayout/data/val_new.txt'
list_path = 'D:/Projects/PanoLayout/PanoLayout_master/exp/full/output/lnet_all_depth_list.txt'

def mae(pred, depth):
    """ Mean Average Error (MAE) """
    return np.absolute(pred - depth).mean()


def rmse(pred, depth):
    """ Root Mean Square Error (RMSE) """
    return math.sqrt(np.power((pred - depth), 2).mean())


def rel(pred, depth):
    """ Mean Absolute Relative Error (REL) """
    return (np.absolute(pred - depth) / depth).mean()


def log10(pred, depth):
    """ Mean log10 Error (LOG10) """
    return np.absolute(np.log10(pred) - np.log10(depth)).mean()


def delta1(pred, depth, delta=1.25):
    """ Threshold delta1 """
    #raise NotImplementedError
    thr = np.maximum(depth/pred, pred/depth)
    return (thr < delta).astype(np.float32).mean()

def delta2(pred, depth, delta=1.25):
    """ Threshold delta2 """
    #raise NotImplementedError
    thr = np.maximum(depth/pred, pred/depth)
    return (thr < delta**2).astype(np.float32).mean()

def delta3(pred, depth, delta=1.25):
    """ Threshold delta2 """
    #raise NotImplementedError
    thr = np.maximum(depth/pred, pred/depth)
    return (thr < delta**3).astype(np.float32).mean()

def gen_path(root_path, pano_id, file_name):

    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]


mae_lst = [[],[],[],[],[]]
rmse_lst = [[],[],[],[],[]]

for i, pano_id in enumerate(id_list):

    gt_path = gen_path(gt_dataset_path, pano_id, 'depth.png')

    if pano_id[0] == 'X':
        print('X')
        continue

    pred_path = os.path.join(dataset_path, "{0}.png".format(i))
    print(pred_path)
    if not os.path.isfile(pred_path):
        continue

    label_path = gen_path(gt_dataset_path, pano_id, 'label.json')

    with open(label_path, 'r') as f:
        jdata = json.load(f)
    
    num = jdata['layoutPoints']['num']
    #print(num)

    gt_img = Image.open(gt_path)
    gt_img = np.array(gt_img, np.float32) / 4000

    pred_img = Image.open(pred_path)
    pred_img = np.array(pred_img, np.float32) / 4000

    #l1 =  np.abs(pred_img - gt_img).sum()

    list_id = int((num-4)/2)
    list_id = list_id if list_id <= 3 else 3
    
    mae_ = mae(pred_img, gt_img)
    print(mae_)

    rmse_ = rmse(pred_img, gt_img)
    print(rmse_)

    mae_lst[list_id].append(mae_)
    rmse_lst[list_id].append(rmse_)  


mae_all = []
rmse_all = []

print('=============')
for i in range(4):
    print('{0} num: {1}'.format( (i*2+4) ,len(mae_lst[i])))
    print(np.mean(mae_lst[i]))
    print(np.mean(rmse_lst[i]))
    mae_all += mae_lst[i]
    rmse_all += rmse_lst[i]
    print('-------------------')

print('Total:')
print(np.mean(mae_all))
print(np.mean(rmse_all))
