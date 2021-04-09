# LED<sup>2</sup>-Net

This is PyTorch implementation of our CVPR 2021 Oral paper "LED<sup>2</sup>-Net: Monocular 360˚ Layout Estimation via Differentiable Depth Rendering". 

**You can visit our project website and upload your own panorama to see the 3D results!**

<a href='https://fuenwang.ml/project/led2net/'>[Project Website]</a>
<a href='https://arxiv.org/abs/2104.00568/'>[Paper (arXiv)]</a>
<p align='center'><image src='src/3Dlayout.png' width='100%'></image></p>

## Prerequisite
This repo is primarily based on <a href='https://pytorch.org/'>PyTorch</a>. You can use the follwoing command to intall the dependencies.
```bash
pip install -r requirements.txt
```

## Preparing Training Data
Under <a href='LED2Net/Dataset'>LED2Net/Dataset</a>, we provide the dataloader of <a href='https://github.com/ericsujw/Matterport3DLayoutAnnotation'>Matterport3D<a> and <a href='https://cgv.cs.nthu.edu.tw/projects/dulanet'>Realtor360<a>. The annotation formats of the two datasets follows <a href='https://github.com/SunDaDenny/PanoAnnotator'>PanoAnnotator</a>. The detailed description of the format is explained in <a href='https://github.com/fuenwang/LayoutMP3D'>LayoutMP3D</a>.

Under <a href='config/'>config/</a>, <a href='config/config_mp3d.yaml'>config_mp3d.yaml</a> and <a href='config/config_realtor360.yaml'>config_realtor360.yaml</a> are the configuration file for Matterport3D and Realtor360.

### Matterport3D
To train/val on Matterport3D, please modify the two items in <a href='config/config_mp3d.yaml'>config_mp3d.yaml</a>.
```yaml
dataset_image_path: &dataset_image_path '/path/to/image/location'
dataset_label_path: &dataset_label_path '/path/to/label/location'
```
The **dataset_image_path** and **dataset_label_path** follow the folder structure:
  ```
    dataset_image_path/
    |-------17DRP5sb8fy/
            |-------00ebbf3782c64d74aaf7dd39cd561175/
                    |-------color.jpg
            |-------352a92fb1f6d4b71b3aafcc74e196234/
                    |-------color.jpg
            .
            .
    |-------gTV8FGcVJC9/
            .
            .
    dataset_label_path/
    |-------mp3d_train.txt
    |-------mp3d_val.txt
    |-------mp3d_test.txt
    |-------label/
            |-------Z6MFQCViBuw_543e6efcc1e24215b18c4060255a9719_label.json
            |-------yqstnuAEVhm_f2eeae1a36f14f6cb7b934efd9becb4d_label.json
            .
            .
            .
  ```
Then run **main.py** and specify the config file path
```bash
python main.py --config config/config_mp3d.yaml --mode train # For training
python main.py --config config/config_mp3d.yaml --mode val # For testing
```

### Realtor360
To train/val on Realtor360, please modify the item in <a href='config/config_realtor360.yaml'>config_realtor360.yaml</a>.
```yaml
dataset_path: &dataset_path '/path/to/dataset/location'
```
The **dataset_path** follows the folder structure:
  ```
    dataset_path/
    |-------train.txt
    |-------val.txt
    |-------sun360/
            |-------pano_ajxqvkaaokwnzs/
                    |-------color.png
                    |-------label.json
            .
            .
    |-------istg/
            |-------1/
                    |-------1/
                            |-------color.png
                            |-------label.json
                    |-------2/
                            |-------color.png
                            |-------label.json
                    .
                    .
            .
            .
            
    
  ```
Then run **main.py** and specify the config file path
```bash
python main.py --config config/config_realtor360.yaml --mode train # For training
python main.py --config config/config_realtor360.yaml --mode val # For testing
```

## Run Inference
After finishing the training, you can use the following command to run inference on your own data (xxx.jpg or xxx.png).
```bash
python run_inference.py --config YOUR_CONFIG --src SRC_FOLDER/ --dst DST_FOLDER --ckpt XXXXX.pkl
```
This script will predict the layouts of all images (jpg or png) under **SRC_FOLDER/** and store the results as json files under **DST_FOLDER/**.

### Pretrained Weights
We provide the pretrained model of <a href='https://cgv.cs.nthu.edu.tw/projects/dulanet'>Realtor360<a> in this <a href='https://drive.google.com/file/d/1cayRqxee8CxKJnaFQXPc6r6ngWjlgciF/view?usp=sharing'>link</a>.

**Currently, we use DuLa-Net's post processing for inference. We will release the version using HorizonNet's post processing later.**

## Layout Visualization
To visualize the 3D layout, we provide the visualization tool in <a href='https://github.com/fuenwang/360LayoutVisualizer'>360LayoutVisualizer</a>. Please clone it and install the corresponding packages. Then, run the following command
```bash
cd 360LayoutVisualizer/
python visualizer.py --img xxxxxx.jpg --json xxxxxx.json
```
<image src='src/visualize.png' width='50%'>
  
## Citation
```bibtex
@misc{wang2021led2net,
      title={LED2-Net: Monocular 360 Layout Estimation via Differentiable Depth Rendering}, 
      author={Fu-En Wang and Yu-Hsuan Yeh and Min Sun and Wei-Chen Chiu and Yi-Hsuan Tsai},
      year={2021},
      eprint={2104.00568},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
