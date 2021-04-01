import math
import numpy as np
import copy
from attrdict import AttrDict

class MovingAverageEstimator(object):
    """ Estimate moving average of the given results """
    def __init__(self, field_names):
        self.field_names = field_names
        self.all_results = []
    
    def update(self, pred, gt, pred_height, gt_height):
        results = AttrDict()
        for field in self.field_names:
            results[field] = globals()[field](pred, gt, pred_height, gt_height)
        self.all_results.append(copy.deepcopy(results))
        return results
    
    def __call__(self):
        total = len(self.all_results)
        out = AttrDict()
        for field in self.field_names:
            val_lst = [one[field] for one in self.all_results]
            out[field] = np.mean(val_lst)
        
        return out

def IoU_2D(pred, gt, dummy_height1=None, dummy_height2=None):
    intersect = np.sum(np.logical_and(pred, gt))
    union = np.sum(np.logical_or(pred, gt))
    iou_2d = intersect / union

    return iou_2d

def IoU_3D(pred, gt, pred_height, gt_height):
    intersect = np.logical_and(pred, gt)
    fp_t_pred = pred.astype(int) - intersect.astype(int)
    fp_t_gt = gt.astype(int) - intersect.astype(int)

    union = fp_t_pred.sum()*pred_height + fp_t_gt.sum()*gt_height + intersect.sum() * max(pred_height, gt_height)
    intersect = intersect.sum() * min(pred_height, gt_height)
    iou_3d = intersect / union

    return iou_3d