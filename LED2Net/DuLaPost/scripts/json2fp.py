import sys
import os
import argparse

import objs
import utils

def json2fp(json, size, ratio):

    scene = objs.Scene()
    utils.loadLabelByJson(json, scene)
    floorMap = utils.genLayoutFloorMap(scene, size, ratio)

    return floorMap

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    args = parser.parse_args()

    labelPath = args.i
    outputPath = os.path.dirname(args.i)

    floorMap = json2fp(labelPath, [1000, 1000], 0.02)
    #utils.saveImage(floorMap, os.path.join(outputPath, 'fp_full.png'))
    utils.showImage(floorMap)