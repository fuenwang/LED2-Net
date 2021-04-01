import cv2
import torch
import numpy as np

__all__ = ['XY2lonlat', 'lonlat2xyz', 'XY2xyz', 'xyz2lonlat', 'lonlat2XY', 'xyz2XY', 'EquirecTransformer']

def XY2lonlat(xy, shape, mode='numpy'):
    lon = ((xy[..., 0] - ((shape[1]-1) / 2.0)) / shape[1]) * 2 * np.pi
    lat = ((xy[..., 1] - ((shape[0]-1) / 2.0)) / shape[0]) * np.pi
    lon = lon[..., None]
    lat = lat[..., None]
    out = np.concatenate([lon, lat], axis=-1) if mode == 'numpy' else torch.cat([lon, lat], dim=-1)

    return out

def lonlat2xyz(lonlat, RADIUS=1.0, mode='numpy'):
    # lonlat is (... x 2)
    lon = lonlat[..., 0:1]
    lat = lonlat[..., 1:]

    cos = np.cos if mode == 'numpy' else torch.cos
    sin = np.sin if mode == 'numpy' else torch.sin

    x = RADIUS * cos(lat) * sin(lon)
    y = RADIUS * sin(lat)
    z = RADIUS * cos(lat) * cos(lon)
    lst = [x, y, z]

    out = np.concatenate(lst, axis=-1) if mode == 'numpy' else torch.cat(lst, dim=-1)
    return out

def XY2xyz(xy, shape, mode='numpy'):
    lonlat = XY2lonlat(xy, shape, mode)
    xyz = lonlat2xyz(lonlat, mode=mode)

    return xyz

def xyz2lonlat(xyz, clip, mode='numpy'):
    atan2 = np.arctan2 if mode == 'numpy' else torch.atan2
    asin = np.arcsin if mode == 'numpy' else torch.asin

    norm = np.linalg.norm(
        xyz, axis=-1) if mode == 'numpy' else torch.norm(xyz, p=2, dim=-1)
    xyz_norm = xyz / norm[..., None]
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    if mode == 'torch':
        if clip:
            lat = asin(torch.clamp(y, -0.99, 0.99))
        else:
            lat = asin(y)
    else:
        if clip:
            lat = asin(np.clip(y, -0.99, 0.99))
        else:
            lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1) if mode == 'numpy' else torch.cat(lst, dim=-1)
    return out

def lonlat2XY(lonlat, shape, mode='numpy'):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1) if mode == 'numpy' else torch.cat(lst, dim=-1)
    return out

def xyz2XY(xyz, shape, clip=False, mode='numpy'):
    lonlat = xyz2lonlat(xyz, clip, mode)
    XY = lonlat2XY(lonlat, shape, mode)

    return XY

class EquirecTransformer:
    def __init__(self, mode, clip=False):
        assert mode in ['numpy', 'torch']
        self.mode = mode
        self.clip = clip

    def XY2lonlat(self, xy, shape=(512, 1024)):
        return XY2lonlat(xy, shape, self.mode)

    def lonlat2xyz(self, lonlat, RADIUS=1.0):
        return lonlat2xyz(lonlat, RADIUS, self.mode)

    def XY2xyz(self, xy, shape=(512, 1024)):
        return XY2xyz(xy, shape, self.mode)

    def xyz2lonlat(self, xyz):
        return xyz2lonlat(xyz, self.clip, self.mode)
        
    def lonlat2XY(self, lonlat, shape=(512, 1024)):
        return lonlat2XY(lonlat, shape, self.mode)

    def xyz2XY(self, xyz, shape=(512, 1024)):
        return xyz2XY(xyz, shape, self.clip, self.mode)