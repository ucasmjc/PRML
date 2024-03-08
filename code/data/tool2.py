import mmcv
import os
from PIL import Image
import numpy as np
img = 'car-kitti/img/train/'
ann='car-kitti/ann/train/'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(img)

n = 0
list=[]
import shutil

# 定义源文件路径和目标文件路径
src_path = 'car-kitti/img/val/'
dst_path = 'car-kitti/img/train/'

# 使用shutil.move()函数转移文件

for i in range(100):
    shutil.move(src_path+str(i)+'.png', dst_path+str(i+255)+'.png')
    shutil.move('car-kitti/ann/val/' + str(i) + '.png', 'car-kitti/ann/train/' + str(i + 255) + '.png')
for i in fileList:
    ann1=np.array(Image.open(ann+i))
    ann1[ann1 != 255] = 1
    ann1[ann1 == 255] = 0
    mmcv.imwrite(mmcv.imresize(ann1,(1242,375),interpolation='nearest'),ann+i)
    mmcv.imwrite(mmcv.imresize(mmcv.imread(img + i), (1242,375)), img + i)
for i in os.listdir('car-kitti/img/val/'):
    ann1=np.array(Image.open('car-kitti/ann/val/'+i))
    ann1[ann1!=255]=1
    ann1[ann1==255]=0
    mmcv.imwrite(mmcv.imresize(ann1,(1242,375),interpolation='nearest'),'car-kitti/ann/val/'+i)
    mmcv.imwrite(mmcv.imresize(mmcv.imread('car-kitti/img/val/'+i), (1242,375)), 'car-kitti/img/val/'+i)
