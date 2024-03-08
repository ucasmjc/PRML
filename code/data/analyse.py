import os.path
import json
import torch
from mmseg.apis import MMSegInferencer,init_model
import mmcv
import numpy as np
from PIL import Image
def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                        num_classes: int):
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0,
        max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    if area_union[1]==0:
        return 'nan'
    return area_intersect[1]/area_union[1]

model='best_aAcc_iter_11280.pth'
config='loccfg/mv3+pappm+cus.py'
img='data/portrait/val/img'
inferencer = MMSegInferencer(model=config,weights=model)

results = inferencer(img,return_datasamples=True)
a=[]
sum=0
for result in results:
    pred = result.pred_sem_seg.data
    res = torch.squeeze(pred).cpu()
    name=(result.img_path.split('\\')[-1]).split('.')[0]
    label='data/portrait/val/ann/'+name+'.png'
    label=torch.from_numpy(np.array(Image.open(label))).cpu()
    iou=intersect_and_union(res,label,2)
    if iou<0.8:
        a.append(iou)
print(a)