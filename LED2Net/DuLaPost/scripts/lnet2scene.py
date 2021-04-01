import sys
sys.path.append('..')
import os
import argparse
import numpy as np

import objs
import utils

def lnet2scene(data_path):

    with open(data_path) as f:
        content = f.readlines()
        data = [x.strip().split() for x in content]

    camera_h = 1.6

    scene = objs.Scene()
    scene.cameraHeight = 1.6
    scene.layoutHeight = float(data[0][0])

    for i in range(1,5):
        xyz = (float(data[i][0]), 0, -float(data[i][1]))
        print(xyz)
        scene.layoutPoints.append(objs.GeoPoint(scene, None, xyz))

    scene.genLayoutWallsByPoints(scene.layoutPoints)
    scene.updateLayoutGeometry()

    return scene
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    parser.add_argument('--gt', required=True)
    args = parser.parse_args()

    data_path = args.i
    
    scene_pred = lnet2scene(data_path)
    fp_pred = utils.genLayoutFloorMap(scene_pred, (512,512), 20/512)

    gt_path = args.gt
    scene_gt = objs.Scene()
    utils.loadLabelByJson(gt_path, scene_gt)
    scene_gt.normalize(cameraH=1.6)
    fp_gt = utils.genLayoutFloorMap(scene_gt, (512,512), 20/512)


    utils.showImage([fp_pred, fp_gt])