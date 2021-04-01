import sys
sys.path.append('..')
import os
import argparse

import objs
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    args = parser.parse_args()

    labelPath = args.i
    outputPath = os.path.dirname(args.i)

    scene = objs.Scene()
    utils.loadLabelByJson(labelPath, scene)
    scene.normalize()

    mapSize = [512, 1024, 3]

    #edgeMap = utils.genLayoutEdgeMap(scene, mapSize)
    #utils.saveImage(edgeMap, os.path.join(outputPath, 'edge.png'))

    #normalMap = utils.genLayoutNormalMap(scene, mapSize)
    #utils.saveImage(normalMap, os.path.join(outputPath, 'normal.png'))

    #depthMap = utils.genLayoutDepthMap(scene, mapSize)
    #utils.saveDepth(depthMap, os.path.join(outputPath, 'depth.png'))

    #obj2dMap = utils.genLayoutObj2dMap(scene, mapSize)
    #utils.saveImage(obj2dMap, os.path.join(outputPath, 'obj2d.png'))
    
    fcMap = utils.genLayoutFloorCeilingMap(scene, [512, 1024])
    utils.saveImage(fcMap, os.path.join(outputPath, 'fcmap.png'))
    
    corMap = utils.genLayoutCornerMap(scene, [512, 1024], dilat=4, blur=20)
    utils.saveImage(corMap, os.path.join(outputPath, 'cor.png'))

    edgMap = utils.genLayoutEdgeMap(scene, [512 , 1024, 3], dilat=4, blur=20)
    utils.saveImage(edgMap, os.path.join(outputPath, 'edg_b.png'))