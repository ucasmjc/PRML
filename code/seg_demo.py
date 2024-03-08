import argparse

from show.optic_flow_process import optic_flow_process
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


from mmseg.apis import MMSegInferencer

def read_frame_sequence_from_folder(folder_path):
    frame_sequence = []

    # 列出文件夹中的所有文件
    filenames = os.listdir(folder_path)

    # 按文件名排序确保顺序正确
    filenames.sort()

    # 逐个读取图像文件并添加到帧序列
    for filename in filenames:
        # 构建图像文件的完整路径
        image_path = os.path.join(folder_path, filename)

        # 使用 OpenCV 读取图像
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 将帧添加到帧序列
        frame_sequence.append(frame)

    return frame_sequence


def use_optic_flow(origin_img, pred_img, score_map, prev_gray, prev_cfd, disflow, is_first_frame):
    cur_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.resize(cur_gray, (pred_img.shape[-1], pred_img.shape[-2]))
    optflow_map = optic_flow_process(cur_gray, score_map, prev_gray, prev_cfd, disflow, is_first_frame)
    prev_gray = cur_gray.copy()
    prev_cfd = optflow_map.copy()
    is_first_frame = False
    score_map = optflow_map / 255.
    return score_map


def read_img_sequence_from_folder(folder_path):
    img_sequence = []

    # 列出文件夹中的所有文件
    filenames = os.listdir(folder_path)

    # 按文件名排序确保顺序正确
    filenames.sort()

    # 逐个读取图像文件并添加到帧序列
    for filename in filenames:
        # 构建图像文件的完整路径
        image_path = os.path.join(folder_path, filename)

        # 使用 OpenCV 读取图像
        frame = cv2.imread(image_path)

        # 将帧添加到帧序列
        img_sequence.append(frame)

    return img_sequence


def post_process(img):
    score_map = img.copy()
    # 进行post process
    mask_original = score_map.copy()
    mask_original = (mask_original).astype("uint8")
    _, mask_thr = cv2.threshold(mask_original, 240, 1, cv2.THRESH_BINARY)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
    mask_erode = cv2.erode(mask_thr, kernel_erode)
    mask_dilate = cv2.dilate(mask_erode, kernel_dilate)

    return mask_dilate * 255


def reverse_transform(pred, target_shape, mode='nearest'):
    # 获取目标图像的形状
    target_h, target_w = target_shape

    # 执行反向操作，例如反向缩放
    pred = F.interpolate(pred, size=(target_h, target_w), mode=mode)

    return pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'this is a description')
    parser.add_argument('--config', '-c', required=False, choices = ['model1', 'model2'], dest='config')
    parser.add_argument('--img_path', '-i', required=False, type=str, dest='img_path')
    parser.add_argument('--bg_img_path', '-b', required=False, type=str, dest='bg_img_path')
    parser.add_argument('--save_dir', '-s', required=False, type=str, dest='save_dir')
    parser.add_argument('--vertical_screen', '-v', required=False, action='store_true', dest='ver')
    args = parser.parse_args()

    if args.config == 'model1':
        checkpoint_path = 'checkpoints/Supervise-Portrait.pth'
        config_path = 'final_config.py'
    elif args.config == 'model2':
        checkpoint_path = 'checkpoints/PP-HumanSeg14K.pth'
        config_path = 'baidu.py'
    elif args.img_path is None:
        checkpoint_path = 'checkpoints/PP-HumanSeg14K.pth'
        config_path = 'baidu.py'

    if args.img_path is not None:
        img_path = args.img_path
    else:
        img_path = 'show/test.jpg'

    #这一行会加载模型，会花一些时间，放在初始化里，后边只要用inferencer就行
    inferencer = MMSegInferencer(model=config_path,weights=checkpoint_path)
    # 推理给定图像
    result = inferencer(img_path, show=False)





    frame_sequence = []  # 存放预测score_map
    img_sequence = []  # 存放原图
    score_map_temprary = []  # 暂时存放score_map
    output_sequence = []  # 存放输出的图片
    post_or_not = True

    frame_sequence = [result["predictions"]]  # 读取从模型中输出的预测图序列
    if args.img_path is not None:
        frame = cv2.imread(args.img_path)
    else:
        frame = cv2.imread('show/test.jpg')
    img_sequence.append(frame)


    origin_img = img_sequence[0]
    h, w, _ = origin_img.shape
    if args.bg_img_path is not None:
        bg = cv2.imread(args.bg_img_path)  # 读取背景图片
    else:
        bg = cv2.imread('show/test_bg.jpg')

    # 读入原始数据
    score_map = frame_sequence[0]


    if post_or_not:
        # 如果输入的为单帧则执行后处理操作。
        for frame in frame_sequence:
            score_map_temprary.append(post_process(frame))


    for score_maps in score_map_temprary:
        score_maps = torch.from_numpy(score_map)

        score_maps = score_maps.unsqueeze(0).unsqueeze(0)

        # 进行逆变换操作
        score_maps = reverse_transform(score_maps.float(), target_shape=(h, w), mode='bilinear')

        alpha = score_maps.squeeze(1).permute(1, 2, 0).numpy()
        # print(alpha)

        bg = cv2.resize(bg, (w, h))

        if bg.ndim == 2:
            bg = bg[..., np.newaxis]

        out = (alpha * origin_img + (1 - alpha) * bg).astype(np.uint8)

        output_sequence.append(out)

    if args.save_dir is not None:
        output_folder = args.save_dir  # 保存输出图像的文件夹路径
    else:
        output_folder = 'output'
    if not os.path.exists(output_folder):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(output_folder)

    n = len(os.listdir(output_folder)) + 1
    filepath = os.path.join(output_folder, 'output' + str(n) +'.jpg')
    cv2.imwrite(filepath, output_sequence[-1])


    if args.ver:
        while True:
            cv2.imshow('frames', output_sequence[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break





