import utils 

class GeoPoint(object):

    def __init__(self, scene, coords=None, xyz=None):

        self.coords = coords
        self.color = (0, 0, 0)
        self.depth = 0
        self.xyz = xyz

        self.type = 0 # [concave, convex, occul]
        self.id = 0

        if self.coords == None:
            self.coords = utils.xyz2coords(self.xyz)

        coordsT = (self.coords[1], self.coords[0])

        cpos = utils.coords2pos(coordsT, scene.color.shape)
        self.color = tuple(scene.color[cpos[0]][cpos[1]])

        dpos = utils.coords2pos(coordsT, scene.depth.shape)
        self.depth = scene.depth[dpos[0]][dpos[1]]

        if self.xyz == None:
            self.xyz = utils.coords2xyz(self.coords, self.depth)