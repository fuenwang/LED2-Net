"""
Definition of evaluation metric. Please modify this code very carefully!!

Notes:
- [CHECK] log10 produces NaN

Last update: 2018/11/07 by Johnson
"""
import math
import numpy as np
from attrdict import AttrDict


class MovingAverageEstimator(object):
    """ Estimate moving average of the given results """
    def __init__(self, field_names, align_median=True):
        self.field_names = field_names
        self.align_median = align_median
        self.metrics = Metrics(field_names)
        self.results = AttrDict()
        self.reset()

    def update(self, pred, depth):
        results = self.metrics.compute(pred, depth, self.align_median)
        for name in self.field_names:
            self.results[name] += results[name]
        self.count += 1

        return results

    def __call__(self):
        avg_results = AttrDict()
        for name in self.field_names:
            avg_results[name] = self.results[name] / self.count

        return avg_results

    def reset(self):
        for name in self.field_names:
            self.results[name] = 0.
        self.count = 0

    def __repr__(self):
        return 'Moving Average Estimator: ' + ', '.join(self.field_names)


class Metrics(object):
    """ Benchmark """
    def __init__(self, field_names):
        """ Metrics to be evaluated are specified in `field_names`.
            Make sure you used metrics are defined in this file. """
        self.metric_fn = AttrDict()
        self.results = AttrDict()
        for name in field_names:
            self.metric_fn[name] = globals()[name]
            self.results[name] = 0.
        self.field_names = field_names

    def compute(self, pred, depth, align_median):
        """ Compute results. Note that `pred` and `depth` are numpy array
            and they should have the same shape. """
        valid_mask = (depth > 0.01) & (depth < 10)
        pred_valid = pred[valid_mask]
        depth_valid = depth[valid_mask]
        if align_median:
            pred_median = np.median(pred_valid)
            depth_median = np.median(depth_valid)
            scale = depth_median / pred_median
            pred_valid *= scale
        for name in self.field_names:
            self.results[name] = self.metric_fn[name](pred_valid, depth_valid)

        return AttrDict(self.results.copy())

    def __repr__(self):
        return 'Metrics: ' + ', '.join(self.field_names)


def mae(pred, depth):
    """ Mean Average Error (MAE) """
    return np.absolute(pred - depth).mean()


def rmse(pred, depth):
    """ Root Mean Square Error (RMSE) """
    return math.sqrt(np.power((pred - depth), 2).mean())

def rmse_log(pred, depth):
    mask = pred > 0.01
    pred = pred.copy()[mask]
    depth = depth.copy()[mask]
    a = np.log10(pred)
    b = np.log10(depth)
    return math.sqrt(np.power((a - b), 2).mean())

def mre(pred, depth):
    """ Mean Absolute Relative Error (MRE) """
    return (np.absolute(pred - depth) / depth).mean()


def log10(pred, depth):
    """ Mean log10 Error (LOG10) """
    return np.absolute(np.log10(pred) - np.log10(depth)).mean()


def delta1(pred, depth, delta=1.25):
    """ Threshold delta1 """
    mask = pred > 0.01
    pred = pred.copy()[mask]
    depth = depth.copy()[mask]

    thr = np.maximum(depth/pred, pred/depth)
    return (thr < delta).astype(np.float32).mean()


def delta2(pred, depth, delta=1.25):
    """ Threshold delta2 """
    mask = pred > 0.01
    pred = pred.copy()[mask]
    depth = depth.copy()[mask]

    thr = np.maximum(depth/pred, pred/depth)
    return (thr < delta**2).astype(np.float32).mean()


def delta3(pred, depth, delta=1.25):
    """ Threshold delta2 """
    mask = pred > 0.01
    pred = pred.copy()[mask]
    depth = depth.copy()[mask]

    thr = np.maximum(depth/pred, pred/depth)
    return (thr < delta**3).astype(np.float32).mean()