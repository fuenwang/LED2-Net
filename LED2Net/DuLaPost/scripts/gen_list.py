import sys
import os
import json


input_path = "D:/Dataset/SUN360/orig/panoContext_testmap.txt"
output_path = "D:/Dataset/SUN360/orig/panoContext_test.txt"

with open(input_path) as f:
    id_list = [x.strip().split() for x in f.readlines()]


with open(output_path, "w+") as f:

    for item in id_list:

        pano_id = os.path.splitext(item[0])[0]
        print(pano_id)

        f.write(pano_id + '\n')
