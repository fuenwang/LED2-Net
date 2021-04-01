import random

import objs
import utils 

class WallPlane(object):

    def __init__(self, scene, gPoints):

        self.scene = scene
        self.attached = []
            
        self.gPoints = gPoints
        self.color = (0, 0, 0)

        self.normal = (0, 0, 0)
        self.planeEquation = (0, 0, 0, 0)
        self.width = 0 

        self.corners = []
        self.edges = []
        self.bbox2d = ((0,0),(1,1))

        self.id = 0

        self.updateGeometry()

    def updateGeometry(self):

        self.updateCorners()
        self.updateEdges()
        self.updateBbox2d()

        self.normal = utils.pointsNormal(self.corners[0].xyz,self.corners[1].xyz,
                                        self.corners[3].xyz)
        self.color = utils.normal2color(self.normal)
        self.planeEquation = utils.planeEquation(self.normal, self.corners[0].xyz)
        self.width =  utils.pointsDistance(self.corners[0].xyz, self.corners[1].xyz)

        for obj2d in self.attached:
            obj2d.updateGeometry()

    def updateCorners(self):

        gps = self.gPoints
        cameraH = self.scene.cameraHeight
        cam2ceilH = self.scene.layoutHeight - cameraH

        self.corners = [objs.GeoPoint(self.scene, None, 
                        (gps[0].xyz[0], cam2ceilH, gps[0].xyz[2])),
                        objs.GeoPoint(self.scene, None, 
                        (gps[1].xyz[0], cam2ceilH, gps[1].xyz[2])),
                        objs.GeoPoint(self.scene, None, 
                        (gps[1].xyz[0], -cameraH, gps[1].xyz[2])),
                        objs.GeoPoint(self.scene, None, 
                        (gps[0].xyz[0], -cameraH, gps[0].xyz[2]))]
    
    def updateEdges(self):

        self.edges = [objs.GeoEdge(self.scene, (self.corners[0], self.corners[1])),
                    objs.GeoEdge(self.scene, (self.corners[1], self.corners[2])),
                    objs.GeoEdge(self.scene, (self.corners[2], self.corners[3])),
                    objs.GeoEdge(self.scene, (self.corners[3], self.corners[0]))]

        self.edges[3].type = self.gPoints[0].type
        self.edges[1].type = self.gPoints[1].type
       
    def updateBbox2d(self):

        coords = []
        for c in [e.coords for e in self.edges]:
            coords += c 
        self.bbox2d = utils.imagePointsBox(coords)

    #manh only
    def checkRayHit(self, vec, orig=(0,0,0)):

        point = utils.vectorPlaneHit(vec, self.planeEquation)
        if point is None:
            return False, None
        
        cs = self.corners
        if cs[2].xyz[1] <= point[1] <= cs[0].xyz[1]:

            p1 = (point[0], cs[0].xyz[1], point[2])
            dis1 = utils.pointsDistance(p1, cs[0].xyz)
            dis2 = utils.pointsDistance(p1, cs[1].xyz)
            dis3 = utils.pointsDistance(cs[0].xyz, cs[1].xyz)

            if dis1 + dis2 <= dis3 * 1.0005:
                return True, point

        return False, None

    def getintersection(self, plane):

        for point in self.gPoints:
            if point in plane.gPoints:
                pc = point
            else:
                p1 = point
        for point in plane.gPoints:
            if not point == pc:
                p2 = point
        return (pc, p1, p2)
