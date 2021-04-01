import os
import sys 
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import torch
import glob
import json
from imageio import imread, imwrite
from tqdm import tqdm
import pathlib
import LED2Net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for LED^2-Net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True, help='config.yaml path')
    parser.add_argument('--src', type=str, required=True, help='The folder that contain *.png or *.jpg')
    parser.add_argument('--dst', type=str, required=True, help='The folder to save the output')
    parser.add_argument('--ckpt', type=str, required=True, help='Your pretrained model location (xxx.pkl)')
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = config['exp_args']['device']
    equi_shape = config['exp_args']['visualizer_args']['equi_shape']
    model = LED2Net.Network(**config['network_args']).to(device)
    params = torch.load(args.ckpt)
    model.load_state_dict(params)
    model.eval()

    tmp = [torch.FloatTensor(x).to(device)[None ,...] 
            for x in LED2Net.Dataset.SharedFunctions.create_grid(equi_shape)
          ]
    _, unit_lonlat, unit_xyz = tmp
    infer_height = LED2Net.PostProcessing.InferHeight()
    visualizer = LED2Net.LayoutVisualizer(**config['exp_args']['visualizer_args'])

    src = args.src
    dst = args.dst
    lst = sorted(glob.glob(src+'/*.png') + glob.glob(src+'/*.jpg'))

    for one in tqdm(lst):
        img = LED2Net.Dataset.SharedFunctions.read_image(one, equi_shape)
        batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].to(device)
        with torch.no_grad(): pred = model(batch)

        pred_lonlat_up = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 0, :, None]], dim=-1)
        pred_lonlat_down = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 1, :, None]], dim=-1)
        pred_ratio = infer_height(pred_lonlat_up, pred_lonlat_down)
        pred_corner_num = torch.zeros(pred.shape[0]).to(device).long() + pred.shape[2]
        pred_xyz_down = LED2Net.Conversion.lonlat2xyz(pred_lonlat_down, mode='torch') 
        scale = config['exp_args']['camera_height'] / pred_xyz_down[..., 1:2]
        pred_xyz_down *= scale
        pred_fp_down = visualizer.plot_fp(pred_xyz_down, pred_corner_num)[0, 0, ...].data.cpu().numpy()

        pred_fp_down_man, pred_fp_down_man_pts = LED2Net.DuLaPost.fit_layout(pred_fp_down)
        ratio = pred_ratio[0].data.cpu().numpy()
        pred_height = (ratio+1) * config['exp_args']['camera_height']
        json_data = LED2Net.XY2json(
                pred_fp_down_man_pts.T[:, ::-1], 
                y=config['exp_args']['camera_height'], 
                h=pred_height
                )

        dst_dir = dst + '/%s'%(one.split('/')[-1])
        pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)

        imwrite(dst_dir+'/color.jpg', (img*255).astype(np.uint8))
        with open(dst_dir+'/pred.json', 'w') as f: f.write(json.dumps(json_data, indent=4)+'\n')
