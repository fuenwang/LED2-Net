import sys
import os
import glob
import json
from PIL import Image
import numpy as np

datasetPath = "D:/Projects/PanoLayout/data"
listFilePath = "D:/Projects/PanoLayout/data/sun360_re_test.txt"

def gen_path(root_path, pano_id, file_name):

    path = root_path
    for x in pano_id:
        path = os.path.join(path, x)
    return os.path.join(path, file_name)

with open(listFilePath) as f:
    content = f.readlines()
idList = [x.strip().split() for x in content]

for panoId in idList:

    #filePath = os.path.join(datasetPath, panoId[0], panoId[1], 'label.json')
    filePath = gen_path(datasetPath, panoId, 'label.json')

    if not os.path.isfile(gen_path(datasetPath, panoId, 'fcmap.png')):

        print(filePath)
        os.system("python json2maps.py --i {0}".format(filePath))

        