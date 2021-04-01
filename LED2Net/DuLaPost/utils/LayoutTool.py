from __future__ import division
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/..')
import utils

def alignManhattan(gps):

    class Edge:
        def __init__(self, axis, p1):
            self.axis = axis
            self.points = [p1]
            self.center = (0, 0, 0)

    n = len(gps)
    if n < 2:
        print('cant align manh world')
        return
    
    #create edges, calculate axis type and contain points
    edges = []
    for i in range(n):

        dist = utils.pointsDirectionPow(gps[i].xyz, gps[(i+1)%n].xyz, 2)
        axis = 0 if dist[0] >= dist[2] else 1

        if len(edges) == 0:
            edges.append(Edge(axis, gps[i]))
        elif not edges[-1].axis == axis:
            edges[-1].points.append(gps[i])
            edges.append(Edge(axis, gps[i]))
        elif edges[-1].axis == axis:
            edges[-1].points.append(gps[i])

    #merge last edge to first if they have same axis
    if edges[0].axis == edges[-1].axis:
        edges[0].points += edges[-1].points
        edges.pop()

    #calculate each edge's center position
    for edge in edges:
        pList = [p.xyz for p in edge.points]
        edge.center = utils.pointsMean(pList)

    #calculate manhattan corner points
    manhPoints = []
    for i in range(len(edges)):
        if edges[i].axis == 0:
            manhPoints.append((edges[i-1].center[0], 0, edges[i].center[2]))
        elif edges[i].axis == 1:
            manhPoints.append((edges[i].center[0], 0, edges[i-1].center[2]))

    return manhPoints

def sortWallsByDist(walls):

    return sorted(walls,
                key=lambda x:utils.pointsDistance((0,0,0), utils.pointsMean([x.corners[0].xyz, x.corners[2].xyz])),
                reverse=True)


def calcLayoutPointType(scene):

    walls = scene.layoutWalls
    for i in range(len(walls)):
        w1 = walls[i]
        w2 = walls[(i+1)%len(walls)]
        pc, p1, p2 = w1.getintersection(w2)
        
        v1 = np.array(utils.pointsDirectionPow(pc.xyz, p1.xyz, 1))[[0,2]]
        v1[v1>0] = 1; v1[v1<0] = -1
        v2 = np.array(utils.pointsDirectionPow(pc.xyz, p2.xyz, 1))[[0,2]]
        v2[v2>0] = 1; v2[v2<0] = -1
        n1 = -np.array(w1.normal)[[0,2]]
        n1[n1>0] = 1; n1[n1<0] = -1
        n2 = -np.array(w2.normal)[[0,2]]
        n2[n2>0] = 1; n2[n2<0] = -1
        
        if (v1+n1)[0]==(v2+n2)[0] and (v1+n1)[1]==(v2+n2)[1]:
            pc.type = 0
        else:
            r = -0.0# -0.25
            if w1.planeEquation[3]>r or w2.planeEquation[3]>r:
                pc.type = 2
            else:
                pc.type = 1

def genWallPolygon2d(size, wall):

    size = (size[1], size[0])
    
    isCrossUp, ul, ur = wall.edges[0].checkCross()
    isCrossDown, dl, dr = wall.edges[2].checkCross()

    polygon = []; vertex = []
    for edge in wall.edges:
        vertex.extend([s for s in edge.sample])
        polygon.extend([utils.coords2pos(c,size) for c in edge.coords])

    if not (isCrossUp or isCrossDown):
        return False, polygon
    else:
        iur = vertex.index(ur); iul = iur + 1
        idr = vertex.index(dr); idl = idr - 1
        
        uh = int((polygon[iur][1] + polygon[iul][1])/2)
        dh = int((polygon[idr][1] + polygon[idl][1])/2)        
        polygon1 = polygon[:iur] + [(size[0],uh), (size[0],dh)] + polygon[idr:]
        polygon2 = [(0,uh)] + polygon[iul:idl] + [(0,dh)]
        return True, (polygon1,polygon2)
    
def genLayoutNormalMap(scene, size):
    
    normalMap = np.zeros(size)
    normalMap[:int(size[0]/2),:] = scene.layoutCeiling.color
    normalMap[int(size[0]/2)+1:,:] = scene.layoutFloor.color

    walls = sortWallsByDist(scene.layoutWalls)
    for wall in walls:
        if wall.planeEquation[3] > 0:
            continue
        isCross, polygon = genWallPolygon2d(size, wall)
        if not isCross:
            utils.imageDrawPolygon(normalMap, polygon, wall.color)
        else:
            utils.imageDrawPolygon(normalMap, polygon[0], wall.color)
            utils.imageDrawPolygon(normalMap, polygon[1], wall.color)
    
    return normalMap

def genLayoutOMap(scene, size):

    oMap = np.zeros(size)
    oMap[:,:,0] = 1
        
    walls = sortWallsByDist(scene.layoutWalls)
    for wall in walls:
        if wall.planeEquation[3] > 0:
            continue
        
        color = utils.normal2ManhColor(wall.normal)
        isCross, polygon = genWallPolygon2d(size, wall)
        if not isCross:
            utils.imageDrawPolygon(oMap, polygon, color)
        else:
            utils.imageDrawPolygon(oMap, polygon[0], color)
            utils.imageDrawPolygon(oMap, polygon[1], color)
    
    return oMap

def genLayoutDepthMap(scene, size):

    depthMap = np.zeros(size)

    for y in range(0, size[0]):
        for x in range(0, size[1]):
            coords = utils.pos2coords((y,x), size)
            coordsT = utils.posTranspose(coords)
            vec =  utils.coords2xyz(coordsT, 1)
            if y <= int(size[0]/2):
                plane = scene.layoutCeiling.planeEquation
            else:
                plane = scene.layoutFloor.planeEquation
            point = utils.vectorPlaneHit(vec, plane)
            depth = 0 if point is None else utils.pointsDistance((0,0,0), point)
            depthMap[y,x] = depth

    for wall in scene.layoutWalls:
        if wall.planeEquation[3] > 0:
            continue
        isCross, polygon = genWallPolygon2d(size, wall)
        if not isCross:
            utils.imageDrawWallDepth(depthMap, polygon, wall)
        else:
            utils.imageDrawWallDepth(depthMap, polygon[0], wall)
            utils.imageDrawWallDepth(depthMap, polygon[1], wall)

    return depthMap

def genLayoutEdgeMap(scene, size, dilat=4, blur=20):

    edgeMap = np.zeros(size)
    sizeT = (size[1],size[0])

    walls = sortWallsByDist(scene.layoutWalls)
    for wall in walls:
        #if wall.planeEquation[3] > 0:
        #    continue
        
        '''
        isCross, polygon = genWallPolygon2d(size, wall)
        if not isCross:
            utils.imageDrawPolygon(edgeMap, polygon, (0,0,0))
        else:
            utils.imageDrawPolygon(edgeMap, polygon[0], (0,0,0))
            utils.imageDrawPolygon(edgeMap, polygon[1], (0,0,0))
        '''

        for i, edge in enumerate(wall.edges):
            #color = utils.normal2ManhColor(edge.vector)
            #color = (1, 1, 1)
            #color = type2Color(edge.type)
            color = idx2Color(i)
            
            for i in range(len(edge.coords)-1):
                isCross, l, r = utils.pointsCrossPano(edge.sample[i],
                                                    edge.sample[i+1])
                if not isCross:
                    pos1 = utils.coords2pos(edge.coords[i], sizeT)
                    pos2 = utils.coords2pos(edge.coords[i+1], sizeT)
                    utils.imageDrawLine(edgeMap, pos1, pos2, color)
                else:
                    lpos = utils.coords2pos(utils.xyz2coords(l), sizeT)
                    rpos = utils.coords2pos(utils.xyz2coords(r), sizeT)
                    ch = int((lpos[1] + rpos[1])/2)
                    utils.imageDrawLine(edgeMap, lpos, (0,ch), color)
                    utils.imageDrawLine(edgeMap, rpos, (sizeT[0],ch), color)
        
    edgeMap = utils.imageDilation(edgeMap, dilat)
    edgeMap = utils.imageGaussianBlur(edgeMap, blur)
    for i in range(size[2]):
      edgeMap[:,:,i] *= (1.0/edgeMap[:,:,i].max())

    return edgeMap

def genLayoutObj2dMap(scene, size):

    obj2dMap = np.zeros(size)

    for obj2d in scene.layoutObject2d:
        isCross, polygon = genWallPolygon2d(size, obj2d)
        if not isCross:
            utils.imageDrawPolygon(obj2dMap, polygon, obj2d.color)
        else:
            utils.imageDrawPolygon(obj2dMap, polygon[0], obj2d.color)
            utils.imageDrawPolygon(obj2dMap, polygon[1], obj2d.color)

    return obj2dMap

def genLayoutFloorMap(scene, size, ratio=0.02):

    floorMap = np.zeros(size)

    polygon = []
    for point in scene.layoutFloor.gPoints:
        xz = np.asarray(point.xyz)[[0,2]] / ratio + size[0]/2
        xz[xz>size[0]] = size[0]
        xz[xz<0] = 0
        polygon.append(tuple(xz))

    utils.imageDrawPolygon(floorMap, polygon)

    return floorMap

def genLayoutCornerMap(scene, size, dilat=4, blur=20):

    corMap = np.zeros(size)

    for point in scene.layoutFloor.corners + scene.layoutCeiling.corners:
        pos = utils.coords2pos(utils.posTranspose(point.coords), size)
        corMap[pos[0]][pos[1]] = 1

    corMap = utils.imageDilation(corMap, dilat)
    corMap = utils.imageGaussianBlur(corMap, blur)
    corMap *= (1.0/corMap.max())

    return corMap

def genLayoutFloorCornerMap(scene, size, ratio=0.02, dilat=4, blur=20):

    corMap = np.zeros(size)
    for point in scene.layoutFloor.gPoints:
        xz = np.asarray(point.xyz)[[0,2]] / ratio + size[0]/2
        xz[xz>=size[0]] = size[0]-1
        xz[xz<0] = 0
        corMap[int(xz[1])][int(xz[0])] = 1

    corMap = utils.imageDilation(corMap, dilat)
    corMap = utils.imageGaussianBlur(corMap, blur)
    corMap *= (1.0/corMap.max())

    return corMap

def genLayoutFloorEdgeMap(scene, size, ratio=0.02, dilat=4, blur=20):

    edgeMap = np.zeros(size)

    fpp = scene.layoutFloor.gPoints
    for i in range(len(fpp)):
        xz1 = np.asarray(fpp[i].xyz)[[0,2]] / ratio + size[0]/2
        xz2 = np.asarray(fpp[(i+1)%len(fpp)].xyz)[[0,2]] / ratio + size[0]/2
        p1 = (int(xz1[0]), int(xz1[1]))
        p2 = (int(xz2[0]), int(xz2[1]))
        utils.imageDrawLine(edgeMap, p1, p2)

    edgeMap = utils.imageDilation(edgeMap, dilat)
    edgeMap = utils.imageGaussianBlur(edgeMap, blur)
    edgeMap *= (1.0/edgeMap.max())

    return edgeMap

def genLayoutFloorCeilingMap(scene, size):

    fcMap = np.ones(size)

    walls = sortWallsByDist(scene.layoutWalls)
    for wall in walls:
        if wall.planeEquation[3] > 0:
            continue

        isCross, polygon = genWallPolygon2d(size, wall)
        if not isCross:
            utils.imageDrawPolygon(fcMap, polygon, 0)
        else:
            utils.imageDrawPolygon(fcMap, polygon[0], 0)
            utils.imageDrawPolygon(fcMap, polygon[1], 0)

    return fcMap

def genLayoutFloorPoints(scene, num=1024):

    walls = scene.layoutWalls
    totalWidth = sum([x.width for x in walls])
    wallPnum = [int(round(x.width/totalWidth*num)) for x in walls]
    wallPnum[-1] += (num-sum(wallPnum))

    floorPs = []

    fpp = scene.layoutFloor.gPoints
    for i in range(len(fpp)):
        p1 = np.asarray(fpp[i].xyz)[[0,2]]
        p2 = np.asarray(fpp[(i+1)%len(fpp)].xyz)[[0,2]]
        if wallPnum[i] > 0:
            vec = (p2 - p1)/wallPnum[i]
        else:
            vec = (p2 - p1)/1

        floorPs.append(p1)
        for j in range(wallPnum[i]-1):
            floorPs.append(p1 + vec * (j+1))
    floorPs = np.clip(floorPs, -10, 10)

    for x in walls:
        iscross, l, r = x.edges[0].checkCross()
        if iscross:
            first = np.asarray(l)[[0,2]]
            dis = [np.linalg.norm(p-first) for p in floorPs]
            idx = dis.index(min(dis))

    floorPs = np.roll(np.asarray(floorPs), num-idx, axis=0)
    floorPs = np.swapaxes(floorPs,0,1)
    
    return floorPs

def normal2ManhColor(normal):
    vec = [abs(e) for e in list(normal)]
    axis = vec.index(max(vec))

    if axis == 0:
        color = (0,0,1)
    elif axis == 1:
        color = (1,0,0)
    elif axis == 2:
        color = (0,1,0)
    
    return color

def type2Color(type):

    if type == 0:
        return(0,0,1)
    elif type == 1:
        return(0,1,0)
    else:
        return(1,0,0)

def idx2Color(idx):

    if idx == 0:
       return(0,1,0)
    elif idx == 1 or idx == 3:
        return(1,0,0)
    else:
        return(0,0,1) 
