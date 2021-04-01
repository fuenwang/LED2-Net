import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import least_squares
from functools import partial
from .Conversion import EquirecTransformer

def errorCalculate(ratio, up_norm, down_norm):
    error = np.abs(ratio * up_norm - down_norm)
    #error = np.abs(up_norm - down_norm / ratio)
    return error

class InferHeight(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = 1
        self.et = EquirecTransformer('torch')

    def lonlat2xyz(self, pred_up, pred_down):
        pred_up_xyz = self.et.lonlat2xyz(pred_up)
        s = -self.scale / pred_up_xyz[..., 1:2]
        pred_up_xyz *= s

        pred_down_xyz = self.et.lonlat2xyz(pred_down)
        s = self.scale / pred_down_xyz[..., 1:2]
        pred_down_xyz *= s

        return pred_up_xyz, pred_down_xyz

    def forward(self, pred_up, pred_down):
        pred_up_xyz, pred_down_xyz = self.lonlat2xyz(pred_up, pred_down)
        pred_up_xz = pred_up_xyz[..., ::2]
        pred_down_xz = pred_down_xyz[..., ::2]

        pred_up_norm = torch.norm(pred_up_xz, p=2, dim=-1)
        pred_down_norm = torch.norm(pred_down_xz, p=2, dim=-1)
        #ratio = (pred_up_norm / pred_down_norm).median(dim=-1)[0]
        #ratio = (pred_up_norm / pred_down_norm).mean(dim=-1)
        ratio = self.lsq_fit(pred_up_xz, pred_down_xz)
        #ratio = 1 / ratio
        
        return ratio
    
    def lsq_fit(self, pred_up_xz, pred_down_xz):
        device = pred_up_xz.device
        pred_up_xz = pred_up_xz.cpu().numpy()
        pred_down_xz = pred_down_xz.cpu().numpy()
        ratio = np.zeros(pred_up_xz.shape[0], dtype=np.float32)
        for i in range(pred_up_xz.shape[0]):
            up_xz = pred_up_xz[i, ...].copy()
            up_norm = np.linalg.norm(up_xz, axis=-1)
            down_xz = pred_down_xz[i, ...].copy()
            down_norm = np.linalg.norm(down_xz, axis=-1)
            init_ratio = 1 / np.mean(up_norm / down_norm, axis=-1)
            error_func = partial(errorCalculate, up_norm=up_norm, down_norm=down_norm)
            ret = least_squares(error_func, init_ratio, verbose=0)
            
            x = ret.x[0]
            ratio[i] = x
        ratio = torch.FloatTensor(ratio).to(device)
        return ratio