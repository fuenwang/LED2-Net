import sys

import os
import glob
import json
from shutil import copyfile

dataset_path = "D:/Projects/PanoLayout/data/"
list_path = 'D:/Projects/PanoLayout/data/all.txt'

output_path = 'D:/tmp/'

def gen_path(root_path, pano_id, file_name):

    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]


for i, pano_id in enumerate(id_list):

    file_path = gen_path(dataset_path, pano_id, 'fcmap.png')
    print(file_path)

    if os.path.isfile(file_path):
        output_dir_path = gen_path(output_path, pano_id, '')
        print(output_dir_path)
        
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        copyfile(file_path, os.path.join(output_dir_path, 'fcmap.png'))