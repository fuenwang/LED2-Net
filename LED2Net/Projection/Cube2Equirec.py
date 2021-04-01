import os
import sys
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.misc as sic

class Cube2Equirec(nn.Module):
    def __init__(self, cube_length, equ_h):
        super().__init__()
        self.cube_length = cube_length
        self.equ_h = equ_h
        equ_w = equ_h * 2
        self.equ_w = equ_w
        theta = (np.arange(equ_w) / (equ_w-1) - 0.5) * 2 *np.pi
        phi = (np.arange(equ_h) / (equ_h-1) - 0.5) * np.pi
        
        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

        planes = np.asarray([
                    [0, 0, 1,  1], # z = -1
                    [0, 1, 0, -1], # y =  1
                    [0, 0, 1, -1], # z =  1
                    [1, 0, 0,  1], # x = -1
                    [1, 0, 0, -1], # x =  1
                    [0, 1, 0,  1]  # y = -1
                ])
        r_lst = np.array([
                [0, 1, 0],
                [0.5, 0, 0],
                [0, 0, 0],
                [0, 0.5, 0],
                [0, -0.5, 0],
                [-0.5, 0, 0]
            ]) * np.pi
        f = cube_length / 2.0
        self.K = np.array([
                [f, 0, (cube_length-1)/2.0],
                [0, f, (cube_length-1)/2.0],
                [0, 0, 1]
            ])
        self.R_lst = [cv2.Rodrigues(x)[0] for x in r_lst]

        self.mask, self.XY = self._intersection(xyz, planes)
    
    def forward(self, x, mode='bilinear'):
        assert mode in ['nearest', 'bilinear']
        assert x.shape[0] % 6 == 0
        equ_count = x.shape[0] // 6
        equi = torch.zeros(equ_count, x.shape[1], self.equ_h, self.equ_w).to(x.device)
        for i in range(6):
            now = x[i::6, ...]
            mask = self.mask[i].to(x.device)
            mask = mask[None, ...].repeat(equ_count, x.shape[1], 1, 1)

            XY = (self.XY[i].to(x.device)[None, None, :, :].repeat(equ_count, 1, 1, 1) / (self.cube_length-1) - 0.5) * 2
            sample = F.grid_sample(now, XY, mode=mode, align_corners=True)[..., 0, :]
            equi[mask] = sample.view(-1)

        return equi

    def _intersection(self, xyz, planes):
        abc = planes[:, :-1]
        
        depth = -planes[:, 3][None, None, ...] / np.dot(xyz, abc.T)
        depth[depth < 0] = np.inf
        arg = np.argmin(depth, axis=-1)
        depth = np.min(depth, axis=-1)


        pts = depth[..., None] * xyz
        
        mask_lst = []
        mapping_XY = []
        for i in range(6):
            mask = arg == i
            mask = np.tile(mask[..., None], [1, 1, 3])

            XY = np.dot(np.dot(pts[mask].reshape([-1, 3]), self.R_lst[i].T), self.K.T)
            XY = np.clip(XY[..., :2].copy() / XY[..., 2:], 0, self.cube_length-1)
            mask_lst.append(mask[..., 0])
            mapping_XY.append(XY)
        mask_lst = [torch.BoolTensor(x) for x in mask_lst]
        mapping_XY = [torch.FloatTensor(x) for x in mapping_XY]

        return mask_lst, mapping_XY




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    batch = torch.zeros(12, 3, 256, 256) + 20
    c2e = Cube2Equirec(256, 512)
    equi = c2e(batch) 

    plt.imshow(equi[0, ...].permute(1, 2, 0).cpu().numpy())
    plt.show()
