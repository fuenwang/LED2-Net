import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from .Projection import Equirec2Cube

class LayoutVisualizer(object):
    def __init__(self, cube_dim, equi_shape, camera_FoV, fp_dim, fp_meters):
        self.fp_dim = fp_dim
        self.fp_meters = fp_meters
        self.FoV = camera_FoV / 180.0 * math.pi
        self.e2c = Equirec2Cube(cube_dim, equi_shape[0], camera_FoV)
        self.r_lst = np.array([
            [0, -180.0, 0],
            [90.0, 0, 0],
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [-90, 0, 0]
        ], np.float32) / 180.0 * np.pi

        self.face_order = ['back', 'down', 'front', 'left', 'right', 'top']
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        f = 0.5 * cube_dim / np.tan(0.5 * self.FoV)
        cx = (cube_dim - 1) / 2
        cy = cx
        self.K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], np.float32)
    

    def plot_layout_to_rgb(self, rgb, xyz, num, view='top'):
        device = rgb.device
        face = self.e2c(rgb)[5::6, ...].permute(0, 2, 3, 1).data.cpu().numpy()
        xyz = xyz.data.cpu().numpy()
        num = num.data.cpu().numpy()

        R = self.R_lst[self.face_order.index(view)]
        K = self.K

        xyz = xyz @ R.T
        xyz_cam = xyz @ K.T
        XY_cam = xyz_cam[..., :-1].copy()
        XY_cam /= xyz_cam[..., 2:]
        XY_cam = np.round(XY_cam).astype(int)

        #print (face.shape)
        bs = face.shape[0]
        out = []
        for b in range(bs):
            img = face[b].copy().astype(np.float32)
            draw_XYs = XY_cam[b, :num[b], ...]
            draw_XYs = np.concatenate([draw_XYs, draw_XYs[0:1, ...]], axis=0)
            for i in range(draw_XYs.shape[0]-1):
                src = tuple(draw_XYs[i])
                dst = tuple(draw_XYs[i+1])
                try:
                    cv2.line(img, src, dst, (0, 1.0, 0), thickness=1)
                except:
                    #print (src, dst)
                    pass
            out.append(img.transpose(2, 0, 1)[None, ...])
        out = np.concatenate(out, axis=0)

        return torch.FloatTensor(out).to(device)
    
    def plot_fp(self, xyz, num):
        device = xyz.device
        xyz = xyz.data.cpu().numpy()
        num = num.data.cpu().numpy()

        fp_all = np.zeros([xyz.shape[0], self.fp_dim , self.fp_dim], np.float32)
        for b in range(xyz.shape[0]):
            xz = xyz[b, :num[b], ::2]
            xz = np.round(xz / (self.fp_meters / float(self.fp_dim))).astype(np.int)
            xz += self.fp_dim // 2
            cv2.fillPoly(fp_all[b, ...], [xz], 1.0)
        
        return torch.FloatTensor(fp_all[:, None, ...]).to(device)
