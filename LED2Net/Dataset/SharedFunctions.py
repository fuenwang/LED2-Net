import os
import sys
import cv2
import json
import numpy as np
from imageio import imread
from ..Conversion import XY2xyz, xyz2XY, xyz2lonlat, lonlat2xyz

__all__ = [
            'gen_path',
            'filter_by_wall_num',
            'read_image',
            'read_label',
            'get_contour_3D_points',
            'plane_to_depth',
            'render_rgb',
            'create_grid'
        ]

def gen_path(root_path, pano_id, file_name):
    path = root_path
    for x in pano_id: path = os.path.join(path, x)
    return os.path.join(path, file_name)

def filter_by_wall_num(rgb_lst, label_lst, wall_types):
    new_rgb = []
    new_label = []
    for i, one in enumerate(label_lst):
        with open(one, 'r') as f: label = json.load(f)
        if label['layoutWalls']['num'] in wall_types:
            new_rgb.append(rgb_lst[i])
            new_label.append(label_lst[i])

    return new_rgb, new_label

def read_image(image_path, shape):
    img = imread(image_path, pilmode='RGB').astype(np.float32) / 255
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_AREA)

    return img

def read_label(label_path, cH):
    with open(label_path, 'r') as f: label = json.load(f)
    scale = cH / label['cameraHeight']
    camera_height = cH
    camera_ceiling_height = scale * label.get('cameraCeilingHeight', label['layoutHeight']-label['cameraHeight'])
    camera_ceiling_height_ori = label.get('cameraCeilingHeight', label['layoutHeight']-label['cameraHeight'])
    layout_height = scale * label['layoutHeight']

    up_down_ratio = camera_ceiling_height_ori / label['cameraHeight']
    
    xyz = [one['xyz'] for one in label['layoutPoints']['points']]
    planes = [one['planeEquation'] for one in label['layoutWalls']['walls']]
    point_idxs = [one['pointsIdx'] for one in label['layoutWalls']['walls']]
    
    R_180 = cv2.Rodrigues(np.array([0, -1*np.pi, 0], np.float32))[0]
    xyz = np.asarray(xyz)
    xyz[:, 0] *= -1
    xyz = xyz.dot(R_180.T)
    planes += [[0, 1, 0, camera_ceiling_height_ori], [0, 1, 0, -label['cameraHeight']]]
    planes = np.asarray(planes)
    planes[:, :3] = planes[:, :3].dot(R_180.T)

    xyz *= scale
    planes[:, 3] *= scale
    
    out = {
        'cameraHeight': camera_height,
        'layoutHeight': layout_height,
        'cameraCeilingHeight': camera_ceiling_height,
        'xyz': xyz,
        'planes': planes,
        'point_idxs': point_idxs
    }
    return out
    

def get_contour_3D_points(xyz, points_idx, ccH):
    pts = np.asarray([xyz[i] for i in points_idx], np.float32).reshape([-1, 3])[::2, :].copy()
    pts[:, 1] = -ccH

    return pts


def plane_to_depth(grid, planes, points, idxs, ch, cch):
    [h, w, _] = grid.shape
    scale_lst = []
    inter_lst = []
    eps = 1e-2
    for i, plane in enumerate(planes):
        s = -plane[3] / np.dot(grid, plane[:3].reshape([3, 1]))
        intersec = s * grid
        inter_lst.append(intersec[:, :, None, :])
        if i <= len(planes) - 3:
            idx = idxs[i]
            rang = np.concatenate([points[idx[0]][None, :], points[idx[1]][None, :]], axis=0)
            mx_x, mn_x = np.max(rang[:, 0]), np.min(rang[:, 0])
            mx_z, mn_z = np.max(rang[:, 2]), np.min(rang[:, 2])
            mask_x = np.logical_and(intersec[:, :, 0] <= mx_x+eps, intersec[:, :, 0] >= mn_x-eps)
            mask_z = np.logical_and(intersec[:, :, 2] <= mx_z+eps, intersec[:, :, 2] >= mn_z-eps)
            mask = 1 - np.logical_and(mask_x, mask_z)
            mask = np.logical_or(mask, s[:, :, 0] < 0)
        else:
            mask = 1 - np.logical_and(intersec[:, :, 1] <= ch+eps, intersec[:, :, 1] >= -cch-eps)
            mask = np.logical_or(mask, s[:, :, 0] < 0)
        s[mask[:, :, None]] = np.inf
        scale_lst.append(s)
    scale = np.concatenate(scale_lst, axis=2)
    inter = np.concatenate(inter_lst, axis=2)
    min_idx = np.argmin(scale, axis=2)
    x, y = np.meshgrid(range(w), range(h))
    depth = scale[y.ravel(), x.ravel(), min_idx.ravel()].reshape([h, w])
    intersec = inter[y.ravel(), x.ravel(), min_idx.ravel(), :].reshape([h, w, 3])
    
    return depth, intersec

def render_rgb(pts, rgb, shape):
    xy = xyz2XY(pts.astype(np.float32), shape, mode='numpy')
    new_rgb = cv2.remap(rgb, xy[..., 0], xy[..., 1], interpolation=cv2.INTER_LINEAR)

    return new_rgb

def create_grid(shape):
    h, w = shape
    X = np.tile(np.arange(w)[None, :, None], (h, 1, 1))
    Y = np.tile(np.arange(h)[:, None, None], (1, w, 1))
    XY = np.concatenate([X, Y], axis=-1)
    xyz = XY2xyz(XY, shape, mode='numpy') 
    
    l = w // 4
    mean_lonlat = np.zeros([l, 2], dtype=np.float32)
    mean_lonlat[:, 1] = 0
    mean_lonlat[:, 0] = ((np.arange(l) / float(l-1)) * 2 * np.pi - np.pi).astype(np.float32)
    mean_xyz = lonlat2xyz(mean_lonlat, mode='numpy')

    return xyz, mean_lonlat, mean_xyz
