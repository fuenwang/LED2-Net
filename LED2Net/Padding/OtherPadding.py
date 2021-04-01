import torch
import torch.nn as nn
import math
import pdb
import numpy as np

import matplotlib.pyplot as plt
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class CustomPad(nn.Module):
    def __init__(self, pad_func):
        super(CustomPad, self).__init__()
        self.pad_func = pad_func

    def forward(self, x):
        return self.pad_func(x)

class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, x):
        return x


class ZeroPad(nn.Module):
    def __init__(self, pad_s):
        super(ZeroPad, self).__init__()
        self.pad_s = pad_s
    
    def forward(self, x):
        x = F.pad(x, (self.pad_s, self.pad_s, self.pad_s, self.pad_s)) 
        return x

