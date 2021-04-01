import os
import sys
import cv2
import argparse
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, transform
from scipy.optimize import minimize
from scipy.optimize import least_squares

import objs
import utils

#fp is in cam-ceil normal, height is in cam-floor normal
def data2scene(fp_points, height):

    # cam-ceiling / cam-floor
    scale = (height - 1.6) / 1.6
    #layout_fp, fp_points = fit_layout(fp, scale=None, max_cor=12)

    size = 512
    ratio = 20/size

    fp_points = fp_points.astype(float)
    fp_points[0] -= size/2
    fp_points[1] -= size/2
    fp_points *= scale
    fp_points[0] += size/2
    fp_points[1] += size/2
    fp_points = fp_points.astype(int)

    scene = objs.Scene()
    scene.cameraHeight = 1.6
    scene.layoutHeight = height

    scene.layoutPoints = []
    for i in range(fp_points.shape[1]):
        fp_xy = (fp_points[:,i] - size/2) * ratio
        xyz = (fp_xy[1], 0, fp_xy[0])
        scene.layoutPoints.append(objs.GeoPoint(scene, None, xyz))
    
    scene.genLayoutWallsByPoints(scene.layoutPoints)
    scene.updateLayoutGeometry()

    return scene

def f1_score(pred, gt):

    TP = np.zeros(gt.shape); FP = np.zeros(gt.shape)
    FN = np.zeros(gt.shape); TN = np.zeros(gt.shape)

    TP[(pred==gt) & (pred == 1)] = 1
    FP[(pred!=gt) & (pred == 1)] = 1
    FN[(pred!=gt) & (gt == 1)] = 1
    TN[(pred==gt) & (pred == 0)] = 1

    TP = np.sum(TP); FP = np.sum(FP)
    FN = np.sum(FN); TN = np.sum(TN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    accuracy = (TP + TN) / (gt.shape[0]*gt.shape[1])
    f1_score = 2 / ((1 / precision) + (1 / recall))

    return f1_score


def fit_layout(data, max_cor=12):

    #find max connective component
    ret, data_thresh = cv2.threshold(data, 0.5, 1,0)
    data_thresh = np.uint8(data_thresh)
    data_img, data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)

    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    # crop data sub as f1 true
    sub_x, sub_y, w, h = cv2.boundingRect(data_cnt[0])
    data_sub = data_thresh[sub_y:sub_y+h,sub_x:sub_x+w]

    pred = np.ones(data_sub.shape)

    st = 0.25
    min_score = 0.1

######
    def loss_ul(x):
        sample = pred.copy()
        sample[0:int(x[0]), 0:int(x[1])] = 0
        return -f1_score(sample, data_sub)

    res = minimize(loss_ul, np.array([h*st, w*st]), method='nelder-mead', 
                bounds=[(0,h),(0,w)], options={'xtol': 1e-8, 'disp': False})
    ul = res.x.astype(int)

######
    def loss_ur(x):
        sample = pred.copy()
        sample[0:int(x[0]), int(x[1]):w] = 0
        return -f1_score(sample, data_sub)

    res = minimize(loss_ur, np.array([h*st, w*(1-st)]), method='nelder-mead', 
                bounds=[(0,h),(0,w)], options={'xtol': 1e-8, 'disp': False})
    ur = res.x.astype(int)

######
    def loss_dr(x):
        sample = pred.copy()
        sample[int(x[0]):h, int(x[1]):w] = 0
        return -f1_score(sample, data_sub)

    res = minimize(loss_dr, np.array([h*(1-st), w*(1-st)]), method='nelder-mead', 
                bounds=[(0,h),(0,w)], options={'xtol': 1e-8, 'disp': False})
    dr = res.x.astype(int)

######
    def loss_dl(x):
        sample = pred.copy()
        sample[int(x[0]):h, 0:int(x[1])] = 0
        return -f1_score(sample, data_sub)
    
    res = minimize(loss_dl, np.array([h*(1-st), w*st]), method='nelder-mead',
                bounds=[(0,h),(0,w)], options={'xtol': 1e-8, 'disp': False})
    dl = res.x.astype(int)

    #print([ul, ur, dr, dl])
    s_ul = ul[0]*ul[1] / np.sum(data_sub)
    s_ur = ur[0]*(w-ur[1]) / np.sum(data_sub)
    s_dr = (h-dr[0])*(w-dr[1]) / np.sum(data_sub)
    s_dl = (h-dl[0])*dl[1] / np.sum(data_sub)

    #print([s_ul, s_ur, s_dr, s_dl])
    sort_idx = list(np.argsort([s_ul, s_ur, s_dr, s_dl])[::-1])

    assert max_cor in [4, 6, 8, 10, 12]
    max_idx = (max_cor-4)/2

    if s_ul > min_score and (sort_idx.index(0) < max_idx):
        pred[0:int(ul[0]), 0:int(ul[1])] = 0
    if s_ur > min_score and (sort_idx.index(1) < max_idx):
        pred[0:int(ur[0]), int(ur[1]):w] = 0
    if s_dr > min_score and (sort_idx.index(2) < max_idx):
        pred[int(dr[0]):h, int(dr[1]):w] = 0 
    if s_dl > min_score and (sort_idx.index(3) < max_idx):
        pred[int(dl[0]):h, 0:int(dl[1])] = 0

    pred = np.uint8(pred)
    pred_img, pred_cnt, pred_heri = cv2.findContours(pred, 1, 3)

    polygon = [(p[0][1], p[0][0]) for p in pred_cnt[0][::-1]]

    Y = np.array([p[0]+sub_y for p in polygon])
    X = np.array([p[1]+sub_x for p in polygon])
    fp_points = np.concatenate( (Y[np.newaxis,:],X[np.newaxis,:]), axis=0)

    layout_fp = np.zeros(data.shape)
    rr, cc = draw.polygon(fp_points[0], fp_points[1])
    rr = np.clip(rr, 0, data.shape[0]-1)
    cc = np.clip(cc, 0, data.shape[1]-1)
    layout_fp[rr,cc] = 1

    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(data_sub)
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(pred)
        plt.show()

    return layout_fp, fp_points


'''
def fit_layout(data, scale=None, max_cor=12):

    ret, data_thresh = cv2.threshold(data, 0.5, 1,0)
    data_thresh = np.uint8(data_thresh)
    data_img, data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)

    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    sub_x,sub_y,w,h = cv2.boundingRect(data_cnt[0])
    data_sub = data_thresh[sub_y:sub_y+h,sub_x:sub_x+w]

    if False:
        data_sub_invert = np.uint8(np.ones(data_sub.shape) - data_sub)
        label_num, labels = cv2.connectedComponents(data_sub_invert)
        for i in range(1, label_num):
            score = np.count_nonzero(labels == i) / (data_sub.shape[0]*data_sub.shape[1])
            if score < 0.05:
                data_sub[labels==i] = 1
    
    #utils.showImage(data_sub)
    
    #img = data_sub[:,:,np.newaxis] * 255
    #data_sub_img = np.concatenate([img, img, img], axis=2)
    
    dp = np.zeros(data_sub.shape)
    score_map = np.zeros(data_sub.shape)
    score_map[data_sub == 1] = -10
    score_map[data_sub == 0] = 1

    size_h = data_sub.shape[0]-1
    size_w = data_sub.shape[1]-1
    
    ul = [(0,0), (size_h, size_w)]
    ur = [(0,size_w), (size_h, 0)]
    dl = [(size_h,0), (0, size_w)]
    dr = [(size_h,size_w), (0, 0)]

    def find_rect_pt(box):
        
        start, end = box[0], box[1]
        vec = np.clip([end[0]-start[0], end[1]-start[1]], -1, 1)

        dp = np.zeros( (data_sub.shape[0], data_sub.shape[1]) )

        for i in np.arange(start[0]+vec[0], end[0], vec[0]):
            for j in np.arange(start[1]+vec[1], end[1], vec[1]):
                dp[i][j] = dp[i-vec[0]][j] + dp[i][j-vec[1]] - dp[i-vec[0]][j-vec[1]] + score_map[i][j]
        
        score = dp.max() / (data_sub.shape[0]*data_sub.shape[1])
        if score <= 0.05:
            return None, 0
        point = np.argwhere(dp.max() == dp)[0]
        return point, score
    

    polygon = []
    p_ul, s_ul = find_rect_pt(ul)
    p_ur, s_ur = find_rect_pt(ur)
    p_dr, s_dr = find_rect_pt(dr)
    p_dl, s_dl = find_rect_pt(dl)
    sort_idx = list(np.argsort([s_ul, s_ur, s_dr, s_dl])[::-1])

    assert max_cor in [4, 6, 8, 10, 12]
    max_idx = (max_cor-4)/2

    if (p_ul is None) or (sort_idx.index(0) >= max_idx) :
        polygon.append(ul[0])
    else:
        polygon += [(p_ul[0],ul[0][1]), tuple(p_ul) ,(ul[0][0], p_ul[1])]  
    if p_ur is None or (sort_idx.index(1) >= max_idx) :
        polygon.append(ur[0])
    else:
        polygon += [(ur[0][0], p_ur[1]), tuple(p_ur) ,(p_ur[0], ur[0][1])]
    if p_dr is None or (sort_idx.index(2) >= max_idx) :
        polygon.append(dr[0])
    else:
        polygon += [(p_dr[0], dr[0][1]), tuple(p_dr) ,(dr[0][0], p_dr[1])]
    if p_dl is None or (sort_idx.index(3) >= max_idx) :
        polygon.append(dl[0])
    else:
        polygon += [(dl[0][0], p_dl[1]), tuple(p_dl) ,(p_dl[0], dl[0][1])]    

    Y = np.array([p[0]+sub_y for p in polygon])
    X = np.array([p[1]+sub_x for p in polygon])
    fp_points = np.concatenate( (Y[np.newaxis,:],X[np.newaxis,:]), axis=0)

    if scale is not None:
        fp_points = fp_points.astype(float)
        fp_points[0] -= data.shape[0]/2
        fp_points[1] -= data.shape[1]/2
        fp_points *= scale
        fp_points[0] += data.shape[0]/2
        fp_points[1] += data.shape[1]/2
        fp_points = fp_points.astype(int)

    layout_fp = np.zeros(data.shape)
    rr, cc = draw.polygon(fp_points[0],fp_points[1])
    rr = np.clip(rr, 0, data.shape[0]-1)
    cc = np.clip(cc, 0, data.shape[1]-1)
    layout_fp[rr,cc] = 1

    return layout_fp, fp_points
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    args = parser.parse_args()

    data_path = args.i
    #for filepath in glob.iglob(data_path + '/*.npy'):
    #for i in range(404):
    for i in [91, 104, 145, 159, 167, 194, 215, 223, 253, 256, 261, 266, 300, 304, 357, 358]:
        filepath = os.path.join(data_path, '{0}.npy'.format(i))
        print(filepath)
        
        data = np.load(filepath, encoding = 'bytes')[()]

        #color = data['color']
        #fp_floor = data['fp_floor']
        fp_pred = data['pred_fp_merge']

        layout_fp, fp_points = fit_layout(fp_pred)
        #print(fp_points)


        if True:
            fig = plt.figure()
            ax3 = fig.add_subplot(2,1,1)
            ax3.imshow(fp_pred)
            ax4 = fig.add_subplot(2,1,2)
            ax4.imshow(layout_fp)
            plt.show()