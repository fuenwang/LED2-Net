import sys
import os
import glob
import json
from PIL import Image
import numpy as np

from shutil import copyfile

dataset_path = 'D:/Projects/PanoLayout/data/'
list_path = 'D:/Projects/PanoLayout/data/val.txt'

#output_path = 'D:/Projects/PanoLayout/data/istg/'

def gen_path(root_path, pano_id, file_name):

    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]

for pano_id in id_list:

    #filePath = os.path.join(datasetPath, panoId[0],panoId[1], 'label.json')
    
    
    f1 = gen_path(dataset_path, pano_id, 'color.png')
    f2 = gen_path(dataset_path, pano_id, 'label.json')
    f3 = gen_path(dataset_path, pano_id, 'depth.png')
    f4 = gen_path(dataset_path, pano_id, 'normal.png')
    f5 = gen_path(dataset_path, pano_id, 'edge.png')
    f6 = gen_path(dataset_path, pano_id, 'fcmap.png')
    f7 = gen_path(dataset_path, pano_id, 'lines.png')
    f8 = gen_path(dataset_path, pano_id, 'obj2d.png')
    f9 = gen_path(dataset_path, pano_id, 'cor.png')
    f10 = gen_path(dataset_path, pano_id, 'edg_b.png')

    #file_list = [f1,f2,f3,f4,f5,f6,f7,f8]
    file_list = [f9, f10]

    #for f in file_list:
    #    if not os.path.isfile(f):
    #        print(f)
    
    if not os.path.isfile(f9):
        print(f2)
        os.system("python json2maps.py --i {0}".format(f2))
    

        