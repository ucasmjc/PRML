import os

import mmcv
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import numpy as np
import cv2
import shutil
colors = [(255,255,255), (192,192,0), (113,193,46), (123,64,132), (77,128,255), (255,255,0), (34,134,136), (0,0,0)]
color2id = dict(zip(colors, range(len(colors))))
def convert_labels(label):
    mask = np.full(label.shape[:2], 3, dtype=np.uint8)
    for k, v in color2id.items():
        mask[cv2.inRange(label, np.array(k) - 20, np.array(k) + 20) == 255] = v
    return mask
ann_path='data/fudan/ann/'
for i in os.listdir('data/fudan/ann'):
    label = np.array(Image.open(ann_path+i))
    mask = convert_labels(label)
    mask[mask != 7] = 0
    mask[mask == 7] = 1
    Image.fromarray(mask).save('data/fudan/ann/'+i.split('.')[0]+'.png')
    os.remove(ann_path+i)
ann=os.listdir('data/fudan/ann')
for i in range(len(ann)):
    os.rename(ann_path+ann[i],ann_path+str(i)+'.png')
os.mkdir(ann_path+'train')
os.mkdir(ann_path+'val')
for i in range(int(0.9*len(ann))):
    shutil.move(ann_path + str(i) + '.png', ann_path+'train/'+str(i)+'.png')
for i in range(int(0.9*len(ann)),len(ann)):
    shutil.move(ann_path + str(i) + '.png', ann_path + 'val/' + str(i) + '.png')

