import sys
import os
import glob
import json
from PIL import Image
import numpy as np

dataset_path = "D:/Projects/PanoLayout/data/"
list_path = 'D:/Projects/PanoLayout/data/sun360_id_list.txt'

def gen_path(root_path, pano_id, file_name):

    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]

list_4 = open(dataset_path+'sun360_4.txt', "w+")
list_6 = open(dataset_path+'sun360_6.txt', "w+")
list_8 = open(dataset_path+'sun360_8.txt', "w+")
list_10 = open(dataset_path+'sun360_10.txt', "w+")
list_more = open(dataset_path+'sun360_more.txt', "w+")

for pano_id in id_list:

    file_path = gen_path(dataset_path, pano_id, 'label.json')
    if os.path.isfile(file_path):
        with open(file_path) as f:
            jdata = json.load(f)
            cor_num = jdata['layoutPoints']['num']

            if cor_num == 4:
                list_4.write("{0} {1}\n".format(pano_id[0],pano_id[1]))
            elif cor_num == 6:
                list_6.write("{0} {1}\n".format(pano_id[0],pano_id[1]))
            elif cor_num == 8:
                list_8.write("{0} {1}\n".format(pano_id[0],pano_id[1]))
            elif cor_num == 10:
                list_10.write("{0} {1}\n".format(pano_id[0],pano_id[1]))
            else:
                list_more.write("{0} {1}\n".format(pano_id[0],pano_id[1]))