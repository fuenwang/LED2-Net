from __future__ import division
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
from PIL import Image

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
    scene.layoutHeight = float(height)

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

    ret, data_thresh = cv2.threshold(data, 0.5, 1,0)
    data_thresh = np.uint8(data_thresh)
    #data_img, data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)
    data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)
    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    sub_x, sub_y, w, h = cv2.boundingRect(data_cnt[0])
    data_sub = data_thresh[sub_y:sub_y+h,sub_x:sub_x+w]


    #data_img, data_cnt, data_heri = cv2.findContours(data_sub, 1, 2)
    data_cnt, data_heri = cv2.findContours(data_sub, 1, 2)
    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    data_cnt = data_cnt[0]
    epsilon = 0.005*cv2.arcLength(data_cnt,True)
    approx = cv2.approxPolyDP(data_cnt, epsilon,True)

    x_lst = [0,]
    y_lst = [0,]
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i+1)%len(approx)][0]

        if (p2[0]-p1[0]) == 0:
            slope = 10
        else:
            slope = abs((p2[1]-p1[1]) / (p2[0]-p1[0]))
        
        if slope <= 1:
            s = int((p1[1] + p2[1])/2)
            y_lst.append(s)
            
        elif slope > 1:
            s = int((p1[0] + p2[0])/2)
            x_lst.append(s)
            
    x_lst.append(data_sub.shape[1])
    y_lst.append(data_sub.shape[0])
    x_lst.sort()
    y_lst.sort()

    diag = math.sqrt(math.pow(data_sub.shape[1],2) +  math.pow(data_sub.shape[0],2))

    def merge_near(lst):
        group = [[0,]]
        for i in range(1, len(lst)):
            if lst[i] - np.mean(group[-1]) < diag * 0.05:
                group[-1].append(lst[i])
            else:
                group.append([lst[i],])
        group = [int(np.mean(x)) for x in group]
        return group

    x_lst = merge_near(x_lst)
    y_lst = merge_near(y_lst)

    #print(x_lst)
    #print(y_lst)

    img = np.zeros((data_sub.shape[0],data_sub.shape[1],3))
    for x in x_lst:
        cv2.line(img,(x,0), (x,data_sub.shape[0]),(0,255,0),1)
    for y in y_lst:
        cv2.line(img,(0,y), (data_sub.shape[1],y),(255,0,0),1)
        
    ans = np.zeros((data_sub.shape[0],data_sub.shape[1]))
    for i in range(len(x_lst)-1):
        for j in range(len(y_lst)-1):
            sample = data_sub[y_lst[j]:y_lst[j+1] , x_lst[i]:x_lst[i+1]]            
            score = sample.mean()
            if score >= 0.5:
                ans[y_lst[j]:y_lst[j+1] , x_lst[i]:x_lst[i+1]] = 1
    
    pred = np.uint8(ans)
    #pred_img, pred_cnt, pred_heri = cv2.findContours(pred, 1, 3)
    pred_cnt, pred_heri = cv2.findContours(pred, 1, 3)

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
        img = np.zeros((data_sub.shape[0],data_sub.shape[1],3))
        
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i+1)%len(approx)][0]
            slope = abs((p2[1]-p1[1]) / (p2[0]-p1[0]))
        
            if slope <= 1:
                cv2.line(img,(p1[0], p1[1]), (p2[0], p2[1]),(255,0,0),1)   
            elif slope > 1:
                cv2.line(img,(p1[0], p1[1]), (p2[0], p2[1]),(0,255,0),1)

        #cv2.drawContours(img, [approx], 0, (,255,0), 1)
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(img)
        #plt.show()
        fig.savefig('D:/CVPR/figure/post/002/contour2', bbox_inches='tight',transparent=True, pad_inches=0)

    if False:
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(layout_fp)
        fig.savefig('D:/CVPR/figure/post/002/layout_fp', bbox_inches='tight',transparent=True, pad_inches=0)

        #plt.show()

    if False:
        fig = plt.figure()
        plt.axis('off')
        ax1 = fig.add_subplot(2,3,1)
        ax1.imshow(data)
        ax2 = fig.add_subplot(2,3,2)
        ax2.imshow(data_thresh)
        ax3 = fig.add_subplot(2,3,3)
        ax3.imshow(data_sub)
        ax4 = fig.add_subplot(2,3,4)
        #data_sub = data_sub[:,:,np.newaxis]
        #ax4.imshow(img + np.concatenate( (data_sub,data_sub,data_sub),axis=2) * 0.25)
        ax4.imshow(img)
        ax5 = fig.add_subplot(2,3,5)
        ax5.imshow(ans)        
        ax6 = fig.add_subplot(2,3,6)
        ax6.imshow(layout_fp)

        plt.show()

    return layout_fp, fp_points

'''
def fit_layout_old(data, max_cor=12):

    #find max connective component
    ret, data_thresh = cv2.threshold(data, 0.5, 1,0)
    data_thresh = np.uint8(data_thresh)
    data_img, data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)

    data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)

    # crop data sub as f1 true
    sub_x, sub_y, w, h = cv2.boundingRect(data_cnt[0])
    data_sub = data_thresh[sub_y:sub_y+h,sub_x:sub_x+w]

    pred = np.ones(data_sub.shape)

    min_score = 0.05

    def optim(corid):

        def loss(x):
            h_, w_ = int(x[0]),int(x[1])
            box = [[0,0,h_,w_], [0,w_,h_,w], [h_,w_,h,w], [h_,0,h,w_]]
            sample = pred.copy()
            sample[box[corid][0]:box[corid][2], 
                   box[corid][1]:box[corid][3]] = 0
            return -f1_score(sample, data_sub)

        res_lst = []
        for st in [0.1, 0.25]:
            stp = [[h*st, w*st],[h*st, w*(1-st)],[h*(1-st), w*(1-st)],[h*(1-st), w*st]]
            res = minimize(loss, np.array(stp[corid]), method='nelder-mead', 
                            options={'xtol': 1e-8, 'disp': False})
            res_lst.append(res)

        res_lst.sort(key=lambda x: x.fun, reverse=False)
        return res_lst[0] 

######

    res = optim(0)
    ul = res.x.astype(int)

    res = optim(1)
    ur = res.x.astype(int)

    res = optim(2)
    dr = res.x.astype(int)
    
    res = optim(3)
    dl = res.x.astype(int)

    print([ul, ur, dr, dl])
    
    s_ul = ul[0]*ul[1] / (w*h)
    s_ur = ur[0]*(w-ur[1]) / (w*h)
    s_dr = (h-dr[0])*(w-dr[1]) / (w*h)
    s_dl = (h-dl[0])*dl[1] / (w*h)

    print([s_ul, s_ur, s_dr, s_dl])
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    args = parser.parse_args()

    data_path = args.i
    #for filepath in glob.iglob(data_path + '/*.npy'):
    #for i in range(404):
    #for i in [91, 104, 145, 159, 167, 194, 215, 223, 253, 256, 261, 266, 300, 304, 357, 358]:
    for i in [261]:
        filepath = os.path.join(data_path, '{0}.npy'.format(i))
        print(filepath)
        
        data = np.load(filepath, encoding = 'bytes')[()]

        #color = data['color']
        #fp_floor = data['fp_floor']
        fp_pred = data['pred_fp_merge']

        layout_fp, fp_points = fit_layout(fp_pred)
        #fit_layout(fp_pred)
        #print(fp_points)
