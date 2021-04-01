import numpy as np
import matplotlib.pyplot as plt
import os

import utils

from skimage import morphology, filters, draw, transform
from PIL import Image

def imageROI(data, lt, rb):

    regionDate = data[lt[0]:rb[0], lt[1]:rb[1]]
    return regionDate

def imageRegionMean(data, center, steps):

    lt, rb = imageRegionBox(center, steps, data.shape)
    roi = imageROI(data, lt, rb)
    mean = np.nanmean(roi)
    return mean

def imageRegionBox(center, steps, size):

    lt = (center[0] - steps[0], center[1] - steps[1])
    rb = (center[0] + steps[0], center[1] + steps[1])

    lt = checkImageBoundary(lt, size)
    rb = checkImageBoundary(rb, size)
    return lt, rb

def imagePointsBox(posList):

    X = [pos[0] for pos in posList]
    Y = [pos[1] for pos in posList]

    lt = (min(X), min(Y))
    rb = (max(X), max(Y))
    return lt, rb

def checkImageBoundary(pos, size):
        
    x = sorted([0, pos[0], size[0]])[1]
    y = sorted([0, pos[1], size[1]])[1]
    return (x, y)

def imageResize(data, size):

    dataR = transform.resize(data, size, mode='constant')
    return dataR

def imageDilation(data, rad):

    ans = np.zeros(data.shape, dtype=np.float)
    if data.ndim >= 3:
        for i in range(data.shape[2]):
            channel = data[:,:,i]
            ans[:,:,i] = morphology.dilation(channel, 
                                morphology.diamond(rad))
    else:
        ans[:,:] = morphology.dilation(data, 
                                morphology.diamond(rad))

    return ans

def imageGaussianBlur(data, sigma):

    ans = np.zeros(data.shape, dtype=np.float)
    if data.ndim >= 3:
        for i in range(data.shape[2]):
            channel = data[:,:,i]
            ans[:,:,i] = filters.gaussian(channel, sigma)
    else:
        ans[:,:] = filters.gaussian(data, sigma)
    return ans

def imagesMSE(data1, data2):

    if not data1.shape == data2.shape:
        print('size error')
    #data1r = transform.resize(data1, size, mode='constant')
    #data2r = transform.resize(data2, size, mode='constant')

    #data1r[data1r==0] = np.nan
    #data2r[data2r==0] = np.nan
    #mse = np.nanmean((data1r - data2r)**2)
    mse = np.mean((data1 - data2)**2)

    return mse
    
def imageDrawLine(data, p1, p2, color=None):

    rr, cc = draw.line(p1[1],p1[0],p2[1],p2[0])
    #rr[rr<0] = 0; rr[rr>=data.shape[1]] = data.shape[1]-1
    #cc[cc<0] = 0; cc[cc>=data.shape[0]] = data.shape[0]-1

    if color:
        draw.set_color(data, [rr,cc], list(color))
    else:
        data[rr,cc] = 1

def imageDrawPolygon(data, points, color=None):

    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])
    rr, cc = draw.polygon(Y,X)
    if isinstance(color, tuple):
        draw.set_color(data, [rr,cc], list(color))
    elif isinstance(color, int):
        data[rr,cc] = color
    else:
        data[rr,cc] = 1

def imageDrawWallDepth(data, polygon, wall):

    size = (data.shape[1], data.shape[0])
    polyx = np.array([p[0] for p in polygon])
    polyy = np.array([p[1] for p in polygon])

    posy, posx = draw.polygon(polyy, polyx)

    for i in range(len(posy)):
        coords = utils.pos2coords((posx[i],posy[i]), size)
        vec =  utils.coords2xyz(coords, 1)

        point = utils.vectorPlaneHit(vec, wall.planeEquation)
        depth = 0 if point is None else utils.pointsDistance((0,0,0), point)
        color = (depth, depth, depth)
        
        preDepth = data[posy[i],posx[i],0]
        if preDepth == 0:
            draw.set_color(data, [posy[i],posx[i]], list(color))
        elif preDepth > depth:
            draw.set_color(data, [posy[i],posx[i]], list(color))

def showImage(image, cmap='viridis'):

    if isinstance(image, list):
        for i, img in enumerate(image):
            plt.figure(i)
            plt.imshow(img, cmap=cmap, vmin=0, vmax=1)
        plt.show()
    else:
        plt.figure()
        plt.imshow(image, cmap=cmap, vmin=0, vmax=1)
        plt.show()

def saveImage(image, path):

    im = Image.fromarray(np.uint8(image*255))
    im.save(path)

def saveDepth(depth, path):

    depth = depth[:,:,0]
    data = np.uint16(depth*4000)

    array_buffer = data.tobytes()
    img = Image.new("I", data.T.shape)
    img.frombytes(array_buffer, 'raw', "I;16")
    img.save(path)

def saveMask(mask, path):

    mask = mask[:,:,0]
    im = Image.fromarray(np.uint8(mask*255))
    im.save(path)