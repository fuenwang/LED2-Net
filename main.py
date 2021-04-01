import os
import sys 
import cv2
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import LED2Net


def train(train_loader, val_loader, model, config):
    device = config['exp_args']['device']
    multi_gpu = config['exp_args']['multi-gpu']

    writer = SummaryWriter(config['exp_args']['exp_path'])
    model = nn.DataParallel(model.to(device), output_device=device) if multi_gpu else model.to(device)
    module = model.module if multi_gpu else model

    optim = getattr(torch.optim, config['optimizer_args']['type'])(
            model.parameters(),
            **config['optimizer_args']['args']
        )
    for epoch in range(config['exp_args']['epoch']):
        print ('Epoch %d/%d'%(epoch, config['exp_args']['epoch']-1))
        train_an_epoch(train_loader, model, optim, writer, epoch, config)
        results = val_an_epoch(val_loader, module, config)
        print (results)
        module.Save(epoch, accuracy=results['down']['IoU_3D'], replace=True)
        for up_down, tmp in results.items():
            for metric, val in tmp.items(): writer.add_scalar('%s/%s'%(up_down.upper(), metric), val, epoch)

    writer.close()

def train_an_epoch(train_loader, model, optim, writer, epoch, config):
    device = config['exp_args']['device']
    model.train()

    visualizer = LED2Net.LayoutVisualizer(**config['exp_args']['visualizer_args'])
    render_loss = LED2Net.Loss.RenderLoss(**config['loss_args'])
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
    #for i, data in enumerate(train_loader):
        rgb = data['rgb'].to(device)
        corner_num = data['wall-num'].to(device)
        ratio = data['ratio'].to(device)
        unit_lonlat = data['unit-lonlat'].to(device)
        unit_xyz = data['unit-xyz'].to(device)
        gt_lonlat = data['pts-lonlat'].to(device)
        
        pred = model(rgb)
        pred_lonlat_up = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 0, :, None]], dim=-1)
        pred_lonlat_down = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 1, :, None]], dim=-1)
        
        render_loss.setGrid(unit_xyz[0, ...][None, None, ...])
        loss_depth_up, loss_depth_down, xyz_lst, depth_lst = render_loss(pred_lonlat_up, pred_lonlat_down, gt_lonlat, corner_num, ratio)
        loss = loss_depth_up + loss_depth_down
        loss_dict = {
                    'up': loss_depth_up,
                    'down': loss_depth_down,
                    'total': loss
                }

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % config['exp_args']['exp_freq'] == 0:
            step = epoch * len(train_loader) + i
            
            [pred_depth_up, pred_depth_down, gt_depth] = [LED2Net.Tools.normalizeDepth(x) for x in depth_lst]
            pred_xyz_up, pred_xyz_down, GT_xyz_up_sparse, GT_xyz_up_dense = xyz_lst
            pred_xyz_down[..., 1:2] = -config['exp_args']['camera_height'] * ratio[..., None, None]
            pred_corner_num = torch.zeros_like(corner_num) + pred.shape[2]
            
            pred_rgb_up = visualizer.plot_layout_to_rgb(rgb, pred_xyz_up, pred_corner_num)
            pred_rgb_down = visualizer.plot_layout_to_rgb(rgb, pred_xyz_down, pred_corner_num)
            gt_rgb = visualizer.plot_layout_to_rgb(rgb, GT_xyz_up_dense, pred_corner_num)

            pred_fp_up = visualizer.plot_fp(pred_xyz_up, pred_corner_num)
            pred_fp_down = visualizer.plot_fp(pred_xyz_down, pred_corner_num)
            gt_fp = visualizer.plot_fp(GT_xyz_up_sparse, corner_num)
            
            for key, val in loss_dict.items(): writer.add_scalar('Loss/%s'%key, val, step)
            rgb = F.interpolate(rgb, scale_factor=0.25, recompute_scale_factor=True)
            writer.add_images('RGB/equi', rgb, step)
            writer.add_images('RGB/pred-up', pred_rgb_up, step)
            writer.add_images('RGB/pred-down', pred_rgb_down, step)
            writer.add_images('RGB/GT', gt_rgb, step)

            writer.add_images('FP/pred-up', pred_fp_up, step)
            writer.add_images('FP/pred-down', pred_fp_down, step)
            writer.add_images('FP/GT', gt_fp, step)

            writer.add_images('Depth/pred-up', pred_depth_up.repeat(1, 1, 100, 1), step)
            writer.add_images('Depth/pred-down', pred_depth_down.repeat(1, 1, 100, 1), step)
            writer.add_images('Depth/GT', gt_depth.repeat(1, 1, 100, 1), step)


def val_an_epoch(val_loader, model, config):
    device = config['exp_args']['device']
    model.eval()
    visualizer = LED2Net.LayoutVisualizer(**config['exp_args']['visualizer_args'])
    render_loss = LED2Net.Loss.RenderLoss(**config['loss_args'])
    infer_height = LED2Net.PostProcessing.InferHeight()
    layout_metrics_up = LED2Net.Metric.LayoutMetrics.MovingAverageEstimator(**config['metric_args'])
    layout_metrics_down = LED2Net.Metric.LayoutMetrics.MovingAverageEstimator(**config['metric_args'])

    for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        rgb = data['rgb'].to(device)
        corner_num = data['wall-num'].to(device)
        ratio = data['ratio'].to(device)
        unit_lonlat = data['unit-lonlat'].to(device)
        unit_xyz = data['unit-xyz'].to(device)
        gt_lonlat = data['pts-lonlat'].to(device)
        
        with torch.no_grad(): pred = model(rgb)
        pred_lonlat_up = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 0, :, None]], dim=-1)
        pred_lonlat_down = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 1, :, None]], dim=-1)
        pred_ratio = infer_height(pred_lonlat_up, pred_lonlat_down)
        
        render_loss.setGrid(unit_xyz[0, ...][None, None, ...])
        loss_depth_up, loss_depth_down, xyz_lst, depth_lst = render_loss(pred_lonlat_up, pred_lonlat_down, gt_lonlat, corner_num, ratio)

        [pred_depth_up, pred_depth_down, gt_depth] = [LED2Net.Tools.normalizeDepth(x) for x in depth_lst]
        pred_xyz_up, pred_xyz_down, GT_xyz_up_sparse, GT_xyz_up_dense = xyz_lst
        pred_xyz_down[..., 1:2] = -config['exp_args']['camera_height'] * ratio[..., None, None]
        pred_corner_num = torch.zeros_like(corner_num) + pred.shape[2]
        
        pred_rgb_up = visualizer.plot_layout_to_rgb(rgb, pred_xyz_up, pred_corner_num)
        pred_rgb_down = visualizer.plot_layout_to_rgb(rgb, pred_xyz_down, pred_corner_num)
        gt_rgb = visualizer.plot_layout_to_rgb(rgb, GT_xyz_up_dense, pred_corner_num)

        pred_fp_up = visualizer.plot_fp(pred_xyz_up, pred_corner_num).data.cpu().numpy()
        pred_fp_down = visualizer.plot_fp(pred_xyz_down, pred_corner_num).data.cpu().numpy()
        gt_fp = visualizer.plot_fp(GT_xyz_up_sparse, corner_num).data.cpu().numpy()
        pred_height = config['exp_args']['camera_height'] * (pred_ratio.data.cpu().numpy() + 1)
        gt_height = config['exp_args']['camera_height'] * (ratio.data.cpu().numpy() + 1)

        layout_metrics_up.update(pred_fp_up, gt_fp, pred_height, gt_height)
        layout_metrics_down.update(pred_fp_down, gt_fp, pred_height, gt_height)
    
    results_up = layout_metrics_up()
    results_down = layout_metrics_down()
    results = {
        'up': results_up,
        'down': results_down
    }
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for LED^2-Net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True, help='config.yaml path')
    parser.add_argument('--mode', default='train', type=str, required=True, choices=['train', 'val'], help='train/val mode')
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    LED2Net.Tools.fixSeed(config['exp_args']['seed'])
    
    dataset_func = getattr(LED2Net.Dataset, config['dataset_args']['type'])
    train_data = dataset_func(**config['dataset_args']['train']).CreateLoader()
    val_data = dataset_func(**config['dataset_args']['val']).CreateLoader()
    
    model = LED2Net.Network(**config['network_args'])
    model.Load()

    if args.mode == 'train':
        train(train_data, val_data, model, config) 
    else:
        model = model.to(config['exp_args']['device'])
        results = val_an_epoch(val_data, model, config)
        print (results)
