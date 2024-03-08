
import os

import shutil
data_root='data/EG1800/'
ann_path=data_root+'Labels/'
img_path=data_root+'Images/'
ann=os.listdir(ann_path)
img=os.listdir(img_path)
os.mkdir(ann_path+'train')
os.mkdir(ann_path+'val')
os.mkdir(img_path+'train')
os.mkdir(img_path+'val')
for i in range(int(0.9*len(img))):
    shutil.move(ann_path + img[i].split('.')[0] + '.png', ann_path+'train/'+img[i].split('.')[0] + '.png')
    shutil.move(img_path + img[i], img_path+'train/'+img[i])
for i in range(int(0.9*len(img)),len(img)):
    shutil.move(ann_path + img[i].split('.')[0] + '.png', ann_path + 'val/' + img[i].split('.')[0] + '.png')
    shutil.move(img_path + img[i], img_path + 'val/' + img[i])

