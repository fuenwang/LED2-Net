import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .. import Conversion

class EquirecRotate(nn.Module):
    def __init__(self, equ_h):
        super().__init__()
        self.equ_h = equ_h
        self.equ_w = equ_h * 2

        X = torch.arange(self.equ_w)[None, :, None].repeat(self.equ_h, 1, 1)
        Y = torch.arange(self.equ_h)[:, None, None].repeat(1, self.equ_w, 1)
        XY = torch.cat([X, Y], dim=-1).unsqueeze(0)
        self.grid = Conversion.XY2xyz(XY, shape=(self.equ_h, self.equ_w), mode='torch')
    
    def forward(self, equi, axis_angle, mode='bilinear'):
        assert mode in ['nearest', 'bilinear']
        grid = self.grid.to(equi.device).repeat(equi.shape[0], 1, 1, 1)
        R = Conversion.angle_axis_to_rotation_matrix(axis_angle)
        xyz = (R[:, None, None, ...] @ grid[..., None]).squeeze(-1)
        XY = Conversion.xyz2lonlat(xyz, clip=False, mode='torch')
        X, Y = torch.unbind(XY, dim=-1)
        XY = torch.cat([(X/math.pi).unsqueeze(-1), (Y/0.5/math.pi).unsqueeze(-1)], dim=-1)
        sample = F.grid_sample(equi, XY, mode=mode, align_corners=True)

        return sample
        