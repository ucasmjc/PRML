import os

import mmcv
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import numpy as np
#img =Image.open('data/train_data/lab_train/T000015.png')
#print(list(img.getdata()))
import shutil
import json
with open('data/portrait/data.json', "r+") as f:
    load_dict = json.load(f)
    for key,value in load_dict.items():
        shutil.move('data/portrait/val/img/'+key+'.jpg','data/portrait/train/img/'+key+'.jpg')
        shutil.move('data/portrait/val/ann/' + key + '.png', 'data/portrait/train/ann/' + key + '.png')






## python tools/test.py config.py work_dirs/secondtrain/epoch_3.pth --format-only --eval-options "imgfile_prefix=work_dirs/results"
## python tools/analyze_logs.py work_dirs/thirdtrain/20230405_044221.log.json --key loss


