import math

import objs
import utils

class Object2D(object):

    def __init__(self, scene, gPoints, wall):

        self.scene = scene
        
        self.gPoints = gPoints
        self.attach = wall
        self.color = (1, 1, 1)
        
        self.normal = (0, 0, 0)
        self.planeEquation = (0, 0, 0, 0)
        self.width = 0

        self.corners = []
        self.edges = []

        self.bbox2d = ((0,0),(1,1))
        self.localBbox2d = ((0,0),(1,1))

        self.init()

    def init(self):
        
        if not self in self.attach.attached:
            self.attach.attached.append(self)

        self.updateGeometry()


    def updateGeometry(self):

        self.updateGeoPoints()
        self.updateCorners()
        self.updateEdges()
        self.updateBbox2d()

        self.normal = utils.pointsNormal(self.corners[0].xyz,self.corners[1].xyz,
                                        self.corners[3].xyz)
        #self.color = utils.normal2color(self.normal)
        self.planeEquation = utils.planeEquation(self.normal, self.corners[0].xyz)
        self.width =  utils.pointsDistance(self.corners[0].xyz, self.corners[1].xyz)

    def updateGeoPoints(self):
        
        gps = self.gPoints
        acs = self.attach.corners

        #make sure the gpoints are left-up and right-down
        dis = [[],[]]
        xyzs = [gps[0].xyz, (gps[1].xyz[0], gps[0].xyz[1], gps[1].xyz[2]),
                gps[1].xyz, (gps[0].xyz[0], gps[1].xyz[1], gps[0].xyz[2])]
        for i in range(2):
            for xyz in xyzs:
                dis[i].append(utils.pointsDistance(xyz, acs[i*2].xyz))
            xyz = xyzs[dis[i].index(min(dis[i]))]
            gps[i] = objs.GeoPoint(self.scene, None, xyz)
        
        # stick to wall boundary
        localBbox2d = []
        for i in range(2):
            xyz = list(gps[i].xyz)
            dis = utils.pointsDirectionPow(acs[i*2].xyz, gps[i].xyz, 2)
            cxz = math.sqrt(dis[0]+dis[2]) / self.attach.width
            cy = math.sqrt(dis[1]) / self.scene.layoutHeight
            if cxz <= 0.03:
                xyz[0] = acs[i*2].xyz[0]
                xyz[2] = acs[i*2].xyz[2]
                cxz = 0
            if cy <= 0.03:
                xyz[1] = acs[i*2].xyz[1]
                cy = 0
            gps[i] = objs.GeoPoint(self.scene, None, tuple(xyz))    
            coord = (cxz, cy) if i==0 else (1-cxz, 1-cy)
            localBbox2d.append(coord)
        self.localBbox2d = tuple(localBbox2d)   
        #print(self.localBbox2d)

    def updateCorners(self):

        gps = self.gPoints
        scene = self.scene

        self.corners = [objs.GeoPoint(scene, None, gps[0].xyz),
                        objs.GeoPoint(scene, None, 
                        (gps[1].xyz[0], gps[0].xyz[1], gps[1].xyz[2])),
                        objs.GeoPoint(scene, None, gps[1].xyz), 
                        objs.GeoPoint(scene, None, 
                        (gps[0].xyz[0], gps[1].xyz[1] , gps[0].xyz[2]))]

    def updateEdges(self):

        scene = self.scene
        self.edges = [objs.GeoEdge(scene, (self.corners[0], self.corners[1])),
                    objs.GeoEdge(scene, (self.corners[1], self.corners[2])),
                    objs.GeoEdge(scene, (self.corners[2], self.corners[3])),
                    objs.GeoEdge(scene, (self.corners[3], self.corners[0]))]


    def updateBbox2d(self):

        coords = []
        for c in [e.coords for e in self.edges]:
            coords += c 
        self.bbox2d = utils.imagePointsBox(coords)


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



        
        
        