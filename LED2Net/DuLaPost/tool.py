from __future__ import division
import sys
import os
import argparse
import numpy as np
import math

import objs
import utils

def json2scene(json):

    scene = objs.Scene()
    utils.loadLabelByJson(json, scene)
    #scene.normalize(cameraH=1.6)
    scene.normalize_ceiling(ccH=1.6)

    return scene

def json2fp(json, size, ratio):

    scene = objs.Scene()
    utils.loadLabelByJson(json, scene)
    scene.normalize(cameraH=1.6)
    floorMap_down = utils.genLayoutFloorMap(scene, size, ratio)

    scene = objs.Scene()
    utils.loadLabelByJson(json, scene)
    scene.normalize_ceiling(ccH=1.6)
    floorMap_up = utils.genLayoutFloorMap(scene, size, ratio)
    
    return floorMap_down, floorMap_up

def lnet2scene(data_path):

    with open(data_path) as f:
        content = f.readlines()
        data = [x.strip().split() for x in content]

    scene = objs.Scene()
    scene.cameraHeight = 1.6
    scene.layoutHeight = float(data[0][0])
    if math.isnan(scene.layoutHeight):
        return None

    for i in range(1,5):
        if math.isnan(float(data[i][0])) or  math.isnan(float(data[i][1])):
            return None

        xyz = (float(data[i][0]), 0, -float(data[i][1]))
        scene.layoutPoints.append(objs.GeoPoint(scene, None, xyz))
    
    scene.layoutPoints.reverse()

    scene.genLayoutWallsByPoints(scene.layoutPoints)
    scene.updateLayoutGeometry()

    return scene


def depth2map(depth, size, ratio):

    num = depth.shape[0]
    plist = [utils.coords2xyz( (float(i)/num,0.5), depth[i]) for i in np.arange(0, num)]
    
    pointMap = np.zeros(size)
    polygon = []
    for p in plist:
        xz = np.asarray(p)[[0,2]] / ratio + size[0]/2
        xz[xz>size[0]] = size[0]
        xz[xz<0] = 0
        polygon.append(tuple(xz))

    utils.imageDrawPolygon(pointMap, polygon)

    '''
    xy = [[p[0] for p in plist], [p[2] for p in plist]]
    xy = np.asarray(xy)

    pointMap = np.zeros(size)
    xy /= ratio 
    xy += size[0]/2
    xy = xy.astype(int)
    pointMap[xy[1,:],xy[0,:]] = 1
    '''

    return pointMap

def resize_crop(img, scale, size):
    
    re_size = int(img.shape[0]*scale)
    img = utils.imageResize(img, (re_size, re_size))

    if size <= re_size:
        pd = int((re_size-size)/2)
        img = img[pd:pd+size,pd:pd+size]
    else:
        new = np.zeros((size,size))
        pd = int((size-re_size)/2)
        new[pd:pd+re_size,pd:pd+re_size] = img[:,:]
        img = new

    return img


