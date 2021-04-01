import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import cv2
from .. import Conversion


class RenderLoss(nn.Module):
    def __init__(self, camera_height=1.6):
        super(RenderLoss, self).__init__()
        assert camera_height > 0
        self.cH = camera_height
        self.grid = None
        self.c2d = Corner2Depth(None)
        self.et = Conversion.EquirecTransformer('torch')

    def setGrid(self, grid):
        self.grid = grid
        self.c2d.setGrid(grid)
    
    def lonlat2xyz_up(self, pred_up, GT_up, up_down_ratio):
        pred_up_xyz = self.et.lonlat2xyz(pred_up)
        GT_up_xyz = self.et.lonlat2xyz(GT_up)

        s = -(self.cH * up_down_ratio[..., None, None]) / pred_up_xyz[..., 1:2].detach()
        pred_up_xyz *= s

        s = -(self.cH * up_down_ratio[..., None, None]) / GT_up_xyz[..., 1:2]
        GT_up_xyz *= s
        
        return pred_up_xyz, GT_up_xyz

    def lonlat2xyz_down(self, pred_down, dummy1=None, dummy2=None):
        pred_down_xyz = self.et.lonlat2xyz(pred_down)
        s = self.cH / pred_down_xyz[..., 1:2].detach()
        pred_down_xyz *= s

        return pred_down_xyz, None


    def forward(self, pred_up, pred_down, GT_up, corner_nums, up_down_ratio):
        # pred_up, pred_down, GT_up is lonlat
        assert self.grid is not None

        pred_up_xyz, GT_up_xyz = self.lonlat2xyz_up(pred_up, GT_up, up_down_ratio)
        pred_down_xyz, _ = self.lonlat2xyz_down(pred_down)

        gt_depth, _ = self.c2d(GT_up_xyz, corner_nums)
        pred_depth_up, _ = self.c2d(pred_up_xyz, corner_nums, mode='fast')
        pred_depth_down, _ = self.c2d(pred_down_xyz, corner_nums, mode='fast')
        
        GT_up_xyz_dense = self.grid[:, 0, ...] * gt_depth.permute(0, 2, 3, 1)[:, 0, :, :]
        GT_up_xyz_dense[..., 1:2] = -(self.cH * up_down_ratio[..., None, None])

        loss_depth_up = F.l1_loss(pred_depth_up, gt_depth)
        loss_depth_down = F.l1_loss(pred_depth_down, gt_depth)

        return loss_depth_up, loss_depth_down, [pred_up_xyz, pred_down_xyz, GT_up_xyz, GT_up_xyz_dense], [pred_depth_up, pred_depth_down, gt_depth]

class Corner2Depth(nn.Module):
    def __init__(self, grid):
        super(Corner2Depth, self).__init__()
        self.grid = grid

    def setGrid(self, grid):
        self.grid = grid

    def forward(self, corners, nums, shift=None, mode='origin'):
        if mode == 'origin': return self.forward_origin(corners, nums, shift)
        else: return self.forward_fast(corners, nums, shift)

    def forward_fast(self, corners, nums, shift=None):
        if shift is not None: raise NotImplementedError
        grid_origin = self.grid.to(corners.device)
        eps = 1e-2
        depth_maps = []
        normal_maps = []

        for i, num in enumerate(nums):
            grid = grid_origin.clone()
            corners_now = corners[i, ...].clone()
            corners_now = torch.cat([corners_now, corners_now[0:1, ...]], dim=0)
            diff = corners_now[1:, ...] - corners_now[:-1, ...]
            vec_yaxis = torch.zeros_like(diff)
            vec_yaxis[..., 1] = 1
            cross_result = torch.cross(diff, vec_yaxis, dim=1)
            d = -torch.sum(cross_result * corners_now[:-1, ...], dim=1, keepdim=True)
            planes = torch.cat([cross_result, d], dim=1)
            scale_all = -planes[:, 3] / torch.matmul(grid, planes[:, :3].T)

            intersec = []
            for idx in range(scale_all.shape[-1]):
                intersec.append((grid * scale_all[..., idx:idx+1]).unsqueeze(-1))
            intersec = torch.cat(intersec, dim=-1)
            a = corners_now[1:, ...]
            b = corners_now[:-1, ...]

            x_cat = torch.cat([a[:, 0:1], b[:, 0:1]], dim=1)
            z_cat = torch.cat([a[:, 2:], b[:, 2:]], dim=1)

            max_x, min_x = torch.max(x_cat, dim=1)[0], torch.min(x_cat, dim=1)[0]
            max_z, min_z = torch.max(z_cat, dim=1)[0], torch.min(z_cat, dim=1)[0]

            mask_x = (intersec[:, :, :, 0, :] <= max_x+eps) & (intersec[:, :, :, 0, :] >= min_x-eps)
            mask_z = (intersec[:, :, :, 2, :] <= max_z+eps) & (intersec[:, :, :, 2, :] >= min_z-eps)
            mask_valid = scale_all > 0
            mask = ~(mask_x & mask_z & mask_valid)
            scale_all[mask] = float('inf')
            
            depth, min_idx = torch.min(scale_all, dim=-1)
            _, h, w = min_idx.shape
            normal = planes[min_idx.view(-1), :3].view(1, h, w, -1)
            
            depth_maps.append(depth)
            normal_maps.append(normal)
        depth_maps = torch.cat(depth_maps, dim=0).unsqueeze(1)
        normal_maps = torch.cat(normal_maps, dim=0)

        return depth_maps, normal_maps


    def forward_origin(self, corners, nums, shift=None):
        # corners is (bs, 12, 3)
        # nums is (bs, )
        # shift is bs x 2 which are x and z shift
        grid_origin = self.grid.to(corners.device)
        eps = 1e-2
        depth_maps = []
        normal_maps = []
        for i, num in enumerate(nums):
            grid = grid_origin.clone()
            corners_now = corners[i, :num, ...].clone()  # num x 3
            if shift is not None:
                corners_now[..., 0] -= shift[i, 0]
                corners_now[..., 2] -= shift[i, 1]
            # equation: ax + by + cz + d = 0
            #
            corners_now = torch.cat(
                [corners_now, corners_now[0:1, ...]], dim=0)
            planes = []
            for j in range(1, corners_now.shape[0]):
                vec_corner = corners_now[j:j+1, ...] - corners_now[j-1:j, ...]
                vec_yaxis = torch.zeros_like(vec_corner)
                vec_yaxis[..., 1] = 1
                cross_result = torch.cross(vec_corner, vec_yaxis)
                # now corss_result is a b c
                cross_result = cross_result / \
                    torch.norm(cross_result, p=2, dim=-1)[..., None]
                d = -torch.sum(cross_result *
                               corners_now[j:j+1, ...], dim=-1)[..., None]
                abcd = torch.cat([cross_result, d], dim=-1)  # abcd is 1, 4
                planes.append(abcd)
            planes = torch.cat(planes, dim=0)  # planes is num x 4
            assert planes.shape[0] == num
            scale_all = -planes[:, 3] / torch.matmul(grid, planes[:, :3].T)
            depth = []
            for j in range(scale_all.shape[-1]):
                scale = scale_all[..., j]
                intersec = scale[..., None] * grid
                a = corners_now[j+1:j+2, :]
                b = corners_now[j:j+1, :]
                rang = torch.cat([a, b], dim=0)
                max_x, min_x = torch.max(rang[:, 0]), torch.min(rang[:, 0])
                max_z, min_z = torch.max(rang[:, 2]), torch.min(rang[:, 2])

                mask_x = (intersec[..., 0] <= max_x +
                          eps) & (intersec[..., 0] >= min_x-eps)
                mask_z = (intersec[..., 2] <= max_z +
                          eps) & (intersec[..., 2] >= min_z-eps)
                mask_valid = scale > 0
                mask = ~ (mask_x & mask_z & mask_valid)
                #print (scale.max(), scale.min())
                # exit()
                scale[mask] = float('inf')
                depth.append(scale[None, ...])
                #import ipdb
                # ipdb.set_trace()

                #print (depth)
                # exit()

            depth = torch.cat(depth, dim=1)
            depth, min_idx = torch.min(depth, dim=1)
            [_, h, w] = min_idx.shape
            normal = planes[min_idx.view(-1), :3].view(-1, h, w, 3)
            normal_maps.append(normal)
            depth_maps.append(depth[None, ...])
        depth_maps = torch.cat(depth_maps, dim=0)
        normal_maps = torch.cat(normal_maps, dim=0)
        return depth_maps, normal_maps


class ShiftSampler(nn.Module):
    def __init__(self, dim=256, down_ratio=0.5):
        super(ShiftSampler, self).__init__()
        self.dim = dim
        self.down_ratio = down_ratio
        self.grid_x, self.grid_z = np.meshgrid(range(dim), range(dim))
    
    def _GetAngle(self, pred):
        [num, _] = pred.shape
        tmp = np.concatenate([pred, pred[0:1, :]], axis=0)
        abs_cos = []


    def forward(self, pred_xyz, pred_corner_num, gt_xyz, gt_corner_num):
        #
        # pred_xyz bs x 12 x 3
        # pred_corner_num bs,
        # gt_xyz bs x 12 x 3
        # gt_corner_num bs,
        #
        device = pred_xyz.device
        out = np.zeros([pred_xyz.shape[0], 2], dtype=np.float32)

        pred_xyz = pred_xyz.data.cpu().numpy() * self.down_ratio
        pred_corner_num = pred_corner_num.data.cpu().numpy()
        gt_xyz = gt_xyz.data.cpu().numpy() * self.down_ratio
        gt_corner_num = gt_corner_num.data.cpu().numpy()
        
        for i in range(pred_xyz.shape[0]):
            # first find boundary (max/min xz)
            max_x1 = pred_xyz[i, :pred_corner_num[i], 0].max()
            max_x2 = gt_xyz[i, :gt_corner_num[i], 0].max()
            min_x1 = pred_xyz[i, :pred_corner_num[i], 0].min()
            min_x2 = gt_xyz[i, :gt_corner_num[i], 0].min()

            max_z1 = pred_xyz[i, :pred_corner_num[i], 2].max()
            max_z2 = gt_xyz[i, :gt_corner_num[i], 2].max()
            min_z1 = pred_xyz[i, :pred_corner_num[i], 2].min()
            min_z2 = gt_xyz[i, :gt_corner_num[i], 2].min()

            max_x = np.max([max_x1, max_x2])
            min_x = np.min([min_x1, min_x2])
            max_z = np.max([max_z1, max_z2])
            min_z = np.min([min_z1, min_z2])

            pred_xyz_now_normalized = pred_xyz[i, :pred_corner_num[i], :].copy()
            self._GetAngle(pred_xyz_now_normalized)

            pred_xyz_now_normalized[:, 0] = (pred_xyz_now_normalized[:, 0] - min_x) / (max_x - min_x)
            pred_xyz_now_normalized[:, 2] = (pred_xyz_now_normalized[:, 2] - min_z) / (max_z - min_z)

            gt_xyz_now_normalized = gt_xyz[i, :gt_corner_num[i], :].copy()
            gt_xyz_now_normalized[:, 0] = (gt_xyz_now_normalized[:, 0] - min_x) / (max_x - min_x)
            gt_xyz_now_normalized[:, 2] = (gt_xyz_now_normalized[:, 2] - min_z) / (max_z - min_z)

            pred_xz_now_normalized = (pred_xyz_now_normalized[..., ::2] * (self.dim - 1)).round().astype(np.int)
            gt_xz_now_normalized = (gt_xyz_now_normalized[..., ::2] * (self.dim - 1)).round().astype(np.int)

            pred_mask = np.zeros([self.dim, self.dim], np.uint8)
            gt_mask = np.zeros([self.dim, self.dim], np.uint8)

            cv2.drawContours(pred_mask, [pred_xz_now_normalized], -1, 255, cv2.FILLED)
            cv2.drawContours(gt_mask, [gt_xz_now_normalized], -1, 255, cv2.FILLED)
            
            mask = np.logical_and(pred_mask.astype(np.bool), gt_mask.astype(np.bool))
            x_valid = self.grid_x[mask]
            z_valid = self.grid_z[mask]
            idx_choice = np.random.choice(range(z_valid.shape[0]))
            if False:
                plt.subplot('311')
                plt.imshow(pred_mask)
                plt.subplot('312')
                plt.imshow(gt_mask)
                plt.subplot('313')
                plt.imshow(mask)
                plt.show()
            x_choose = x_valid[idx_choice].astype(np.float32)
            z_choose = z_valid[idx_choice].astype(np.float32)
            out[i, 0] = (x_choose / (self.dim - 1)) * (max_x - min_x) + min_x
            out[i, 1] = (z_choose / (self.dim - 1)) * (max_z - min_z) + min_z

        return torch.FloatTensor(out).to(device)
