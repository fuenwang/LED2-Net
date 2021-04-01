import sys
import os
import json


dataset_path = 'D:/Projects/PanoLayout/data'
list_path = 'D:/Projects/PanoLayout/data/all.txt'

def gen_path(root_path, pano_id, file_name):
    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(list_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]

count = [0, 0, 0, 0, 0, 0, 0, 0, 0]# 4,6,8,10,12

for pano_id in id_list:

    file_path = gen_path(dataset_path, pano_id, 'label.json')

    with open(file_path) as jfile:
        jdata = json.load(jfile)
        print(file_path)
        cor_num = jdata['layoutPoints']['num']
        

        count[int((cor_num-4)/2)] += 1

print(count)
    
        
