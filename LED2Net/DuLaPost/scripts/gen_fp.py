import sys
sys.path.append('..')
import os
import json

import objs
import utils

dataset_path = 'D:/Projects/PanoLayout/data/'
list_path = 'D:/Projects/PanoLayout/data/sun360_istg_id_list.txt'

def gen_path(root_path, pano_id, file_name):
    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]

for pano_id in id_list:

    file_path = gen_path(dataset_path, pano_id, 'label.json')
    print(file_path)
    os.system("python ../json2fp.py --i " + file_path)
