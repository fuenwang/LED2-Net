import cv2
import json
import numpy as np
from .Conversion import EquirecTransformer


def XY2json(XY, y, h, dim=512, meters=20.0):
    xz = XY.astype(np.float32)
    xz -= dim // 2
    xz *= (meters / float(dim))
    y_val = np.ones([xz.shape[0], 1]) * y
    xyz = np.concatenate([xz[..., 0:1], y_val, xz[..., 1:]], axis=-1)
    
    return xyz2json(xyz, h)

def xyz2json(xyz, h):
    data = {
            'cameraHeight': 1.6,
            'layoutHeight': h,
            'cameraCeilingHeight': h-1.6,
            'layoutObj2ds': {
                    'num': 0,
                    'obj2ds': []
            },
            'layoutPoints':{
                'num': xyz.shape[0],
                'points': []
            },
            'layoutWalls':{
                'num': xyz.shape[0],
                'walls': []
            }
        }

    xyz = np.concatenate([xyz, xyz[0:1, :]], axis=0)
    R_180 = cv2.Rodrigues(np.array([0, -1*np.pi, 0], np.float32))[0]
    for i in range(xyz.shape[0]-1):
        a = np.dot(R_180, xyz[i, :])
        a[0] *= -1
        b = np.dot(R_180, xyz[i+1, :])
        b[0] *= -1
        c = a.copy()
        c[1] = 0
        normal = np.cross(a-b, a-c)
        normal /= np.linalg.norm(normal)
        d = -np.sum(normal * a)
        plane = np.asarray([normal[0], normal[1], normal[2], d])

        data['layoutPoints']['points'].append({'xyz': a.tolist(), 'id':i})
        
        next_i = 0 if i+1 >= (xyz.shape[0]-1) else i+1
        tmp = {
                'normal': normal.tolist(),
                'planeEquation': plane.tolist(),
                'pointsIdx': [i, next_i]
            }
        data['layoutWalls']['walls'].append(tmp)
    #### Now we need to reorder the layoutPoints to make cullface and earcut work well

    data['layoutPoints']['points'] = [data['layoutPoints']['points'][0]] + data['layoutPoints']['points'][::-1][:-1]

    return data

def interpolate(a, b, count=100):
    x = np.linspace(a[0], b[0], count)[:, None]
    y = np.linspace(a[1], b[1], count)[:, None]
    z = np.linspace(a[2], b[2], count)[:, None]
    xyz = np.concatenate([x, y, z], axis=-1)

    return xyz

def plotXY(rgb, XY, color):
    for i in range(XY.shape[0]-1):
        a = XY[i, ...].round().astype(int)
        b = XY[i+1, ...].round().astype(int)
        if abs(a[0] - b[0]) > 0.5 * rgb.shape[1]:
            continue
        else:
            cv2.line(rgb, tuple(a), tuple(b), color=color, thickness=5)

def json2boundary(rgb, data, color=(0, 0, 255), pts=500):
    R_180 = cv2.Rodrigues(np.array([0, -1*np.pi, 0], np.float32))[0]
    rgb = rgb.copy()
    xyz = np.asarray([x['xyz'] for x in data['layoutPoints']['points']])
    xyz[:, 0] *= -1
    xyz = np.dot(xyz, R_180.T)
    ET = EquirecTransformer('numpy')
    cameraCeilingHeight = data['layoutHeight'] - data['cameraHeight']
    for wall in data['layoutWalls']['walls']:
        idx = wall['pointsIdx']

        a = xyz[idx[0], :].copy()
        a[1] = -cameraCeilingHeight
        b = xyz[idx[1], :].copy()
        b[1] = -cameraCeilingHeight
        ceiling_xyz = interpolate(a, b, count=pts)
        ceiling_XY = ET.xyz2XY(ceiling_xyz, shape=rgb.shape[:2])
        plotXY(rgb, ceiling_XY, color)


        c = xyz[idx[0], :].copy()
        c[1] = data['cameraHeight']
        d = xyz[idx[1], :].copy()
        d[1] = data['cameraHeight']
        floor_xyz = interpolate(c, d)
        floor_XY = ET.xyz2XY(floor_xyz, shape=rgb.shape[:2])
        plotXY(rgb, floor_XY, color)
    return rgb

