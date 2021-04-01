import os
import sys
import cv2
import time
from imageio import imread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Equirec2Cube(nn.Module):
    def __init__(self, cube_dim, equ_h, FoV=90.0):
        super().__init__()
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = np.array([
            [0, -180.0, 0],
            [90.0, 0, 0],
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [-90, 0, 0]
        ], np.float32) / 180.0 * np.pi
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        self._getCubeGrid()

    def _getCubeGrid(self):
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        self.grids = []
        for R in self.R_lst:
            tmp = xyz @ R # Don't need to transpose since we are doing it for points not for camera
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            self.grids.append(torch.FloatTensor(lonlat[None, ...]))
    
    def forward(self, batch, mode='bilinear'):
        [_, _, h, w] = batch.shape
        assert h == self.equ_h and w == self.equ_w
        assert mode in ['nearest', 'bilinear']

        out = []
        for grid in self.grids:
            grid = grid.to(batch.device)
            grid = grid.repeat(batch.shape[0], 1, 1, 1)
            sample = F.grid_sample(batch, grid, mode=mode, align_corners=True)
            out.append(sample)
        out = torch.cat(out, dim=0)
        final_out = []
        for i in range(batch.shape[0]):
            final_out.append(out[i::batch.shape[0], ...])
        final_out = torch.cat(final_out, dim=0)
        return final_out




if __name__ == '__main__':
    img1 = imread('./0_color.png', pilmode='RGB').astype(np.float32) / 255
    img2 = imread('./10_color.png', pilmode='RGB').astype(np.float32) / 255
    img = [img1, img2]
    batch1 = torch.FloatTensor(img1.transpose(2, 0, 1)[None, ...]).cuda()
    batch2 = torch.FloatTensor(img2.transpose(2, 0, 1)[None, ...]).cuda()
    batch = torch.cat([batch1, batch2], dim=0)
    e2c = Equirec2Cube(256, 512)

    cube = e2c(batch).cpu().numpy()
    print (cube.shape)
    face_name = ['back', 'down', 'front', 'left', 'right', 'top']
    import matplotlib.pyplot as plt
    for c in range(cube.shape[0] // 6):
        plt.figure()
        plt.imshow(img[c])
        plt.figure()
        for i in range(6):
            face = cube[c*6+i, ...].transpose(1, 2, 0)
            plt.subplot(2, 3, i+1)
            plt.title(face_name[i])
            plt.imshow(face)
        plt.show()

    