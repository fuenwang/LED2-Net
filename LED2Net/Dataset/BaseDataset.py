import os
import sys
import cv2
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader


class BaseDataset(TorchDataset):
    def __init__(self, **kwargs):
        self.loader_args = kwargs['loader_args']

    def __len__(self,):
        return len(self.data)

    def CreateLoader(self):
        return TorchDataLoader(self, **self.loader_args)
