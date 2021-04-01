import objs
import utils 

class FloorPlane(object):

    def __init__(self, scene, isCeiling=False):

        self.scene = scene

        self.isCeiling = isCeiling

        self.gPoints = scene.layoutPoints
        self.walls = scene.layoutWalls
        self.color = (0,0,0)
        
        self.normal = (0, -1, 0) if isCeiling else (0, 1, 0)
        self.height = 0
        self.planeEquation = (0, 0, 0, 0)

        self.corners = []
        self.edges = []
        self.bbox2d = ((0,0),(1,1))

        self.id = 0

        self.updateGeometry()
    
    def updateGeometry(self):

        cameraH = self.scene.cameraHeight
        cam2ceilH =  self.scene.layoutHeight - cameraH

        self.height = cam2ceilH if self.isCeiling else cameraH 
        self.planeEquation = self.normal + (self.height,)
        self.color = utils.normal2color(self.normal)

        self.updateCorners()
        self.updateEdges()
        self.updateBbox2d()
        
    def updateCorners(self):

        self.corners = []
        for gp in self.gPoints:
            if self.isCeiling:
                xyz = (gp.xyz[0], self.height, gp.xyz[2])
            else:
                xyz = (gp.xyz[0], -self.height, gp.xyz[2])
            corner = objs.GeoPoint(self.scene, None, xyz)
            self.corners.append(corner)
    
    def updateEdges(self):
        
        self.edges = []
        cnum = len(self.corners)
        for i in range(cnum):
            edge = objs.GeoEdge(self.scene, 
                                (self.corners[i], self.corners[(i+1)%cnum]))
            self.edges.append(edge)
    
    def updateBbox2d(self):

        coords = []
        for c in [e.coords for e in self.edges]:
            coords += c 
        self.bbox2d = utils.imagePointsBox(coords)

