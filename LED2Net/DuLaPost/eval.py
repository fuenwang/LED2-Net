from __future__ import division
import sys
import os
import argparse
import numpy as np
from scipy import ndimage

import objs
import utils


def eval_2d_iou(fp_pred, fp_gt):

    intersect = np.sum(np.logical_and(fp_pred, fp_gt))
    union = np.sum(np.logical_or(fp_pred, fp_gt))

    iou_2d = intersect / union

    return iou_2d

def eval_3d_iou(fp_pred, h_pred, fp_gt, h_gt):


    intersect = np.logical_and(fp_pred, fp_gt)
    
    fp_t_pred = fp_pred.astype(int) - intersect.astype(int)
    fp_t_gt = fp_gt.astype(int) - intersect.astype(int)

    union = fp_t_pred.sum()*h_pred + fp_t_gt.sum()*h_gt + intersect.sum()*max(h_pred,h_gt)
    intersect = intersect.sum()*min(h_pred,h_gt)

    iou_3d = intersect / union

    return iou_3d


def eval_l2(pred, gt):

    return np.sqrt(np.mean(((pred - gt)**2)))


def shift_CoM(fp_pred, fp_gt):

    com_pred = ndimage.measurements.center_of_mass(fp_pred)
    com_gt = ndimage.measurements.center_of_mass(fp_gt)

    roll_h =  int(fp_pred.shape[0]/2 - com_pred[0])
    fp_pred_com = np.roll(fp_pred, roll_h, axis=0)
    roll_w =  int(fp_pred.shape[1]/2 - com_pred[1])
    fp_pred_com = np.roll(fp_pred_com, roll_w, axis=1)

    roll_h =  int(fp_gt.shape[0]/2 - com_gt[0])
    fp_gt_com = np.roll(fp_gt, roll_h, axis=0)
    roll_w =  int(fp_gt.shape[1]/2 - com_gt[1])
    fp_gt_com = np.roll(fp_gt_com, roll_w, axis=1)

    return fp_pred_com, fp_gt_com
