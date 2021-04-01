import torch
import random
import numpy as np

def fixSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multiGPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def normalizeDepth(depth):
    d = depth.clone()
    for i in range(depth.shape[0]):
        d[i ,...] -= d[i ,...].min()
        d[i, ...] /= d[i, ...].max()

    return d