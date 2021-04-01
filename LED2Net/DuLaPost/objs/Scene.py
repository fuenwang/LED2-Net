import os
import numpy as np

import objs
import utils

class Scene(object):

    def __init__(self):

        self.color = np.zeros([100,100,3])
        self.depth = np.zeros([100,100])
        
        self.layoutPoints = []
        self.layoutWalls = []
        self.layoutObjects2d = []

        self.layoutFloor = None
        self.layoutCeiling = None

        self.cameraHeight = 1.6
        self.layoutHeight = 3.2

    '''
    def genManhLayout(self):

        num = self.depth.shape[1]
        plist = [objs.GeoPoint(self, (x/num, 0.5)) for x in np.arange(0, num)]

        self.calcManhLayoutPoints(plist)
        self.genLayoutWallsByPoints(self.layoutPoints)
        self.updateLayoutGeometry()

    def calcManhLayoutPoints(self, points):

        manhxyz = utils.alignManhattan(points)
        for xyz in manhxyz:
            gp = objs.GeoPoint(self, None, xyz)
            self.layoutPoints.append(gp)
    '''

    def genLayoutWallsByPoints(self, points):

        pnum = len(points)
        for i in range(0, pnum):
            plane = objs.WallPlane(self, [points[i], points[(i+1)%pnum]])
            self.layoutWalls.append(plane)
       
        self.layoutFloor = objs.FloorPlane(self, False)
        self.layoutCeiling = objs.FloorPlane(self, True)

    def updateLayoutGeometry(self):

        utils.calcLayoutPointType(self)
        for wall in self.layoutWalls:
            wall.updateGeometry()
        self.layoutFloor.updateGeometry()
        self.layoutCeiling.updateGeometry()

    def normalize(self, cameraH=1.6):

        scale = cameraH / self.cameraHeight
        for point in self.layoutPoints:
            point.xyz = utils.vectorMultiplyC(point.xyz, scale)
        
        self.layoutHeight *= scale
        self.cameraHeight = cameraH

        self.updateLayoutGeometry()

    def normalize_ceiling(self, ccH=1.6):

        scale = ccH / (self.layoutHeight-self.cameraHeight)
        for point in self.layoutPoints:
            point.xyz = utils.vectorMultiplyC(point.xyz, scale)
        
        self.cameraHeight *= scale
        self.layoutHeight *= scale

        self.updateLayoutGeometry()
        
    def loadLabel(self, path):
        utils.loadLabelByJson(path, self)
  
    def getPanoName(self):
        return 'nothing'

    def getPanoColorData(self):
        return self.color
    def getPanoDepthData(self):
        return self.depth
    

