import sys
import os
import glob
import json
from PIL import Image
import numpy as np


dataset_path = "D:/Projects/PanoLayout/LayoutNet/result/sun360_istg_4/"
gt_dataset_path = "D:/Projects/PanoLayout/data/"

list_path = 'D:/Projects/PanoLayout/data/val_tmp.txt'

def gen_path(root_path, pano_id, file_name):

    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]
    

cor_l2_list = [[],[],[],[]] 
#edg_l2_list = []

for pano_id in id_list:

    cor_pred_path = gen_path(dataset_path, pano_id, 'edg.png')
    cor_gt_path = gen_path(gt_dataset_path, pano_id, 'edg_b.png')

    if not os.path.isfile(cor_pred_path):
        print(cor_pred_path)
        continue
    if not os.path.isfile(cor_gt_path):
        print(cor_gt_path)
        continue

    with open(gen_path(gt_dataset_path, pano_id, 'label.json')) as f:
        jdata = json.load(f)
        num = jdata['layoutPoints']['num']

    print(cor_pred_path)

    cor_pred = Image.open(cor_pred_path)
    cor_pred = np.array(cor_pred, np.float32) / 255

    cor_gt = Image.open(cor_gt_path)
    cor_gt = np.array(cor_gt, np.float32) / 255

    cor_l2 = np.sqrt(np.mean(((cor_pred - cor_gt)**2)))
    print(cor_l2)

    list_id = int((num-4)/2)
    list_id = list_id if list_id <= 3 else 3
    cor_l2_list[list_id].append(cor_l2)


cor_l2_all = []

print('=============')
for i in range(4):
    print('{0} num: {1}'.format( (i*2+4) ,len(cor_l2_list[i])))
    print(np.mean(cor_l2_list[i]))
    cor_l2_all += cor_l2_list[i]
    print('-------------------')

print('Total:')
print(np.mean(cor_l2_all))