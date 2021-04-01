import torch
import torch.nn as nn
from .. import Conversion

class EquirecGrid(object):
    def __init__(self):
        super().__init__()
        self.bag = {}
        self.ET = Conversion.EquirecTransformer('torch')

    def _createGrid(self, key, h, w):
        X = torch.arange(w)[None, :, None].repeat(h, 1, 1)
        Y = torch.arange(h)[:, None, None].repeat(1, w, 1)
        XY = torch.cat([X, Y], dim=-1).unsqueeze(0)
        self.bag[key] = XY

    def _checkBag(self, h, w):
        key = '(%d,%d)'%(h, w)
        if key not in self.bag: self._createGrid(key, h, w)

        return self.bag[key]


    def to_xyz(self, depth):
        assert len(depth.shape) == 4 and depth.shape[1] == 1
        h, w = depth.shape[2:]
        grid = self._checkBag(h, w).to(depth.device)

        xyz = self.ET.XY2xyz(grid, shape=[h, w]).permute(0, 3, 1, 2) * depth

        return xyz