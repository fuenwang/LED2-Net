import os
import sys
import cv2
import json
import numpy as np
from imageio import imread
import torch
from torch.utils.data import Dataset as TorchDataset
from .BaseDataset import BaseDataset
from ..Conversion import XY2xyz, xyz2XY, xyz2lonlat, lonlat2xyz
from .SharedFunctions import *


class Matterport3DDataset(BaseDataset):
    def __init__(self, dataset_image_path, dataset_label_path, mode, shape, image_name, wall_types, aug, camera_height, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.aug = aug
        self.camera_height = camera_height
        self.grid, self.unit_lonlat, self.unit_xyz = create_grid(shape)
        self.max_wall_num = max(wall_types)

        with open('%s/mp3d_%s.txt'%(dataset_label_path, mode), 'r') as f: lst = [x.rstrip().split() for x in f]
        rgb_lst = [gen_path(dataset_image_path, x, image_name) for x in lst]
        label_lst = ['%s/label/%s_%s_label.json'%(dataset_label_path, *x,) for x in lst]
        rgb_lst, label_lst = filter_by_wall_num(rgb_lst, label_lst, wall_types)
        self.data = list(zip(rgb_lst, label_lst))

    def __getitem__(self, idx):
        rgb_path, label_path = self.data[idx]
        label = read_label(label_path, self.camera_height)
        pts = get_contour_3D_points(label['xyz'], label['point_idxs'], label['cameraCeilingHeight'])
        aug = self.aug
        
        rgb = read_image(rgb_path, self.shape)
        if aug['stretch']:
        #if True:
            kx = np.random.uniform(1, 2)
            kx = 1/kx if np.random.randint(2) == 0 else kx
            ky = np.random.uniform(1, 2)
            ky = 1/ky if np.random.randint(2) == 0 else ky
            kz = np.random.uniform(1, 2)
            kz = 1/kz if np.random.randint(2) == 0 else kz

            d, inter = plane_to_depth(self.grid, label['planes'], label['xyz'], label['point_idxs'], label['cameraHeight'], label['cameraCeilingHeight'])
            inter[:, :, 0] *= kx
            inter[:, :, 1] *= ky
            inter[:, :, 2] *= kz
            

            rgb = render_rgb(inter, rgb, self.shape)
            pts[..., 0] /= kx
            pts[..., 1] /= ky
            pts[..., 2] /= kz

        if aug['rotate']:
            dx = np.random.randint(rgb.shape[1])
            rgb = np.roll(rgb, dx, axis=1)
            angle = (float(dx) / rgb.shape[1]) * 2 * np.pi
            r = cv2.Rodrigues(angle * np.array([0, 1, 0], np.float32))[0]
            pts = np.dot(pts, r.T)

        if aug['flip'] and np.random.randint(2) == 0:
            rgb = np.flip(rgb, axis=1).copy()
            pts[:, 0] *= -1
            pts = pts[::-1, :].copy()

        if aug['gamma']:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0: p = 1 / p
            rgb = rgb ** p

        pts = xyz2lonlat(pts, clip=False, mode='numpy')
        min_x_index = np.argmin(pts[:, 0])
        pts = np.concatenate([pts[min_x_index:, :], pts[:min_x_index, :]], axis=0)
        num = pts.shape[0]
        new_pts = np.zeros([self.max_wall_num, 2], np.float32) + 10000
        new_pts[:num, :] = pts


        rgb = rgb.transpose(2, 0, 1)
        out = {
                'rgb': rgb,
                'pts-lonlat': new_pts,
                'unit-lonlat': self.unit_lonlat,
                'unit-xyz': self.unit_xyz,
                'wall-num': num,
                'ratio': label['cameraCeilingHeight'] / label['cameraHeight'],
                'cameraCeilingHeight': label['cameraCeilingHeight'],
                'location': rgb_path
        }

        return out













