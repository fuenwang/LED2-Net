import sys
import os
import glob
import json
from PIL import Image
import numpy as np

import objs
import utils

import tool
import layout

dataset_path = "D:/Projects/PanoLayout/LayoutNet/result/sun360_istg_4/"
gt_dataset_path = "D:/Projects/PanoLayout/data/"

list_path = 'D:/Projects/PanoLayout/data/val_new.txt'


def eval_2d_iou(fp_pred, fp_gt):

    intersect = np.sum(np.logical_and(fp_pred, fp_gt))
    union = np.sum(np.logical_or(fp_pred, fp_gt))

    iou_2d = intersect / union

    return iou_2d

def eval_3d_iou(fp_pred, h_pred, fp_gt, h_gt):


    intersect = np.logical_and(fp_pred, fp_gt)
    
    fp_t_pred = fp_pred - intersect
    fp_t_gt = fp_gt - intersect

    union = fp_t_pred.sum()*h_pred + fp_t_gt.sum()*h_gt + intersect.sum()*max(h_pred,h_gt)
    intersect = intersect.sum()*min(h_pred,h_gt)

    iou_3d = intersect / union

    return iou_3d


def gen_path(root_path, pano_id, file_name):

    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]


iou_2d_list = [[],[],[],[]] 
iou_3d_list = [[],[],[],[]]

for i, pano_id in enumerate(id_list):

    pred_file_path = gen_path(dataset_path, pano_id, 'lnet_4_post_noalign.txt')
    gt_file_path = gen_path(gt_dataset_path, pano_id, 'label.json')

    if not os.path.isfile(pred_file_path):
        print(pred_file_path)
        continue
    if not os.path.isfile(gt_file_path):
        print(gt_file_path)
        continue

    with open(gt_file_path) as f:
        jdata = json.load(f)
        num = jdata['layoutPoints']['num']

    print(pano_id)

    scene_pred = tool.lnet2scene(pred_file_path)
    if scene_pred is None:
        print('FUCK')
        continue

    if False:
        scene_pred.normalize()
        output_path = os.path.join('D:/Projects/PanoLayout/LayoutNet/result/json_4_noalign','{0}.json'.format(i))
        utils.saveSceneAsJson(output_path, scene_pred)

    fp_pred = utils.genLayoutFloorMap(scene_pred, (512,512), 20/512)

    gt_path = gt_file_path
    scene_gt = objs.Scene()
    utils.loadLabelByJson(gt_path, scene_gt)
    scene_gt.normalize(cameraH=1.6)
    fp_gt = utils.genLayoutFloorMap(scene_gt, (512,512), 20/512)
    #utils.showImage([fp_pred, fp_gt])

    iou_2d = eval_2d_iou(fp_pred, fp_gt)
    #print(iou_2d)
    iou_3d = eval_3d_iou(fp_pred, scene_pred.layoutHeight, fp_gt, scene_gt.layoutHeight)
    #print(iou_3d)

    list_id = int((num-4)/2)
    list_id = list_id if list_id <= 3 else 3
    
    iou_2d_list[list_id].append(iou_2d)
    iou_3d_list[list_id].append(iou_3d)        


iou_2d_all = []
iou_3d_all = []


print('=============')
for i in range(4):
    print('{0} num: {1}'.format( (i*2+4) ,len(iou_2d_list[i])))
    print(np.mean(iou_2d_list[i]))
    print(np.mean(iou_3d_list[i]))
    iou_2d_all += iou_2d_list[i]
    iou_3d_all += iou_3d_list[i]
    print('-------------------')

print('Total:')
print(np.mean(iou_2d_all))
print(np.mean(iou_3d_all))


#DEPTH
'''
f = open(os.path.join(dataset_path, 'lnet_all_depth_list.txt'), "w+")

for i, pano_id in enumerate(id_list):

    pred_file_path = gen_path(dataset_path, pano_id, 'lnet_all_post.txt')
    scene_pred = tool.lnet2scene(pred_file_path)

    if scene_pred is None:
        f.write('X'+ '\n')
        continue

    flag = True
    for wall in scene_pred.layoutWalls:
        if wall.planeEquation[3] >= 0:
            flag = False
        #print(wall.planeEquation)

    if flag:
        line = ''
        for x in pano_id:
            line = line + ' ' + x
        f.write(line+ '\n')
    else:
        f.write(' X'+ '\n')
             
    scene_pred.normalize()
    depth_map = utils.genLayoutDepthMap(scene_pred, (512, 1024,3))
    output_path = gen_path(dataset_path, pano_id, 'depth_all_post.png')
    print(output_path)
    utils.saveDepth(depth_map, output_path)

'''