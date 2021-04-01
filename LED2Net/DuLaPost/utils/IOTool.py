import json
import io
import os
from PIL import Image

import utils
import objs

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def saveSceneAsMaps(path, scene, size):
    
    edgeMap = utils.genLayoutEdgeMap(scene, size)
    utils.saveImage(edgeMap, path + '/edge.png')
    
    oMap = utils.genLayoutOMap(scene, size)
    utils.saveImage(oMap, path + '/label_omap.png')

    normalMap = utils.genLayoutNormalMap(scene, size)
    utils.saveImage(normalMap, path + '/normal.png')

    depthMap = utils.genLayoutDepthMap(scene, size)
    utils.saveDepth(depthMap, path + '/depth.png')

    '''
    obj2dMap = utils.genLayoutObj2dMap(scene, size)
    utils.saveImage(obj2dMap, path + '/obj2d.png')
    '''

def saveSceneAsJson(path, scene):

    points =  scene.layoutPoints
    pointsList = []
    for i, point in enumerate(points):
        pointDict = {
            'coords':point.coords,
            #'depth':float(point.depth),
            'xyz':list(point.xyz),
            'id':point.id
        }
        pointsList.append(pointDict)
    pointsDict = {'num':len(points),
                  'points':pointsList}

    walls = scene.layoutWalls
    wallsList = []
    for i, wall in enumerate(walls):
        wallDict = {
            'pointsIdx':[points.index(wall.gPoints[0]),
                      points.index(wall.gPoints[1])],
            'normal':list(wall.normal),
            'planeEquation':list(wall.planeEquation),
            'width': wall.width,
            'id' : wall.id
        }
        wallsList.append(wallDict)
    wallsDict = {'num':len(walls),
                 'walls':wallsList}

    obj2ds = scene.layoutObjects2d
    obj2dsList = []
    for i, obj2d in enumerate(obj2ds):
        obj2dDict = {
            'wallIdx':walls.index(obj2d.attach),
            'points':[gp.xyz for gp in obj2d.gPoints],
            'coords':[list(obj2d.localBbox2d[0]),
                      list(obj2d.localBbox2d[1])],
            'width': obj2d.width,
            'id' : obj2d.id,
        }
        obj2dsList.append(obj2dDict)
    obj2dsDict = {'num':len(obj2ds),
                 'obj2ds':obj2dsList}

    data = {
            'panoId': scene.getPanoName(),
            'layoutHeight': scene.layoutHeight,
            'cameraHeight': scene.cameraHeight,
            'cameraCeilingHeight': scene.layoutHeight - scene.cameraHeight,
            'layoutPoints':pointsDict,
            'layoutWalls':wallsDict,
            'layoutObj2ds':obj2dsDict}
    
    with io.open(path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                        indent=4, sort_keys=True,
                         ensure_ascii=False)
        outfile.write(to_unicode(str_))

def scene2json(scene):

    points =  scene.layoutPoints
    pointsList = []
    for i, point in enumerate(points):
        pointDict = {
            'coords':point.coords,
            #'depth':float(point.depth),
            'xyz':list(point.xyz),
            'id':point.id
        }
        pointsList.append(pointDict)
    pointsDict = {'num':len(points),
                  'points':pointsList}

    walls = scene.layoutWalls
    wallsList = []
    for i, wall in enumerate(walls):
        wallDict = {
            'pointsIdx':[points.index(wall.gPoints[0]),
                      points.index(wall.gPoints[1])],
            'normal':list(wall.normal),
            'planeEquation':list(wall.planeEquation),
            'width': wall.width,
            'id' : wall.id
        }
        wallsList.append(wallDict)
    wallsDict = {'num':len(walls),
                 'walls':wallsList}

    obj2ds = scene.layoutObjects2d
    obj2dsList = []
    for i, obj2d in enumerate(obj2ds):
        obj2dDict = {
            'wallIdx':walls.index(obj2d.attach),
            'points':[gp.xyz for gp in obj2d.gPoints],
            'coords':[list(obj2d.localBbox2d[0]),
                      list(obj2d.localBbox2d[1])],
            'width': obj2d.width,
            'id' : obj2d.id,
        }
        obj2dsList.append(obj2dDict)
    obj2dsDict = {'num':len(obj2ds),
                 'obj2ds':obj2dsList}

    data = {
            'panoId': scene.getPanoName(),
            'layoutHeight': scene.layoutHeight,
            'cameraHeight': scene.cameraHeight,
            'cameraCeilingHeight': scene.layoutHeight - scene.cameraHeight,
            'layoutPoints':pointsDict,
            'layoutWalls':wallsDict,
            'layoutObj2ds':obj2dsDict}
    
    return data

def loadLabelByJson(path, scene):

    with open(path) as f:
        jdata = json.load(f)

    scene.cameraHeight = jdata['cameraHeight']
    scene.layoutHeight = jdata['layoutHeight']

    pointsDict = jdata['layoutPoints']
    pointsList = pointsDict['points']

    gPoints = []
    for point in pointsList:
        xyz = tuple(point['xyz'])
        gPoint = objs.GeoPoint(scene, None, xyz)
        gPoints.append(gPoint)

    scene.layoutPoints = gPoints
    scene.genLayoutWallsByPoints(scene.layoutPoints)
    scene.updateLayoutGeometry()

    walls = scene.layoutWalls

    if 'layoutObj2ds' in jdata:

        obj2dsDict = jdata['layoutObj2ds']
        obj2dsList = obj2dsDict['obj2ds']

        object2ds = []
        for obj2d in obj2dsList:
            gp1 = objs.GeoPoint(scene, None, tuple(obj2d['points'][0]))
            gp2 = objs.GeoPoint(scene, None, tuple(obj2d['points'][1]))
            wall = walls[int(obj2d['wallIdx'])]
            object2d = objs.Object2D(scene, [gp1, gp2], wall)
            object2ds.append(object2d)
        
        scene.layoutObject2d = object2ds
