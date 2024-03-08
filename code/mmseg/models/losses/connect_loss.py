# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from mmseg.registry import MODELS
import cv2


@MODELS.register_module()
class ConnectLoss(nn.Module):
    """Boundary loss.

    This function is modified from
    `PIDNet <https://github.com/XuJiacong/PIDNet/blob/main/utils/criterion.py#L122>`_.  # noqa
    Licensed under the MIT License.


    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self, loss_weight: float = 1.0,
                 loss_name: str = 'loss_connection',ignore_index=255, max_pred_num_conn=10, use_argmax=True):
        '''
        Args:
            ignore_index (int): Specify a pixel value to be ignored in the annotated image and does not contribute to
                the input gradient.When there are pixels that cannot be marked (or difficult to be marked) in the marked
                image, they can be marked as a specific gray value. When calculating the loss value, the pixel corresponding
                to the original image will not be used as the independent variable of the loss function. *Default:``255``*
            max_pred_num_conn (int): Maximum number of predicted connected components. At the beginning of training,
                there will be a large number of connected components, and the calculation is very time-consuming.
                Therefore, it is necessary to limit the maximum number of predicted connected components,
                and the rest will not participate in the calculation.
            use_argmax (bool): Whether to use argmax for logits.
        '''
        super().__init__()
        self.ignore_index = ignore_index
        self.max_pred_num_conn = max_pred_num_conn
        self.use_argmax = use_argmax
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    def forward(self, logits, labels,**kwargs):
        '''
        Args:
            logits (Tensor): [N, C, H, W]
            lables (Tensor): [N, H, W]
        '''
        preds = torch.argmax(logits, dim=1) if self.use_argmax else logits
        preds_np = preds.cpu().numpy().astype('uint8')
        labels_np = labels.cpu().numpy().astype('uint8')
        multi_class_sc_loss = torch.zeros([preds.shape[0]])
        zero = torch.tensor([0])
        # Traverse each image
        for i in range(preds.shape[0]):
            sc_loss = 0
            class_num = 0

            pred_i = preds[i]
            preds_np_i = preds_np[i]
            labels_np_i = labels_np[i]

            # Traverse each class
            for class_ in np.unique(labels_np_i):
                if class_ == self.ignore_index:
                    continue
                class_num += 1

                # Connected Components Calculation
                preds_np_class = preds_np_i == class_
                labels_np_class = labels_np_i == class_
                pred_num_conn, pred_conn = cv2.connectedComponents(
                    preds_np_class.astype(np.uint8))  # pred_conn.shape = [H,W]
                label_num_conn, label_conn = cv2.connectedComponents(
                    labels_np_class.astype(np.uint8))

                origin_pred_num_conn = pred_num_conn
                if pred_num_conn > 2 * label_num_conn:
                    pred_num_conn = min(pred_num_conn, self.max_pred_num_conn)
                real_pred_num = pred_num_conn - 1
                real_label_num = label_num_conn - 1

                # Connected Components Matching and SC Loss Calculation
                if real_label_num > 0 and real_pred_num > 0:
                    img_connectivity = compute_class_connectiveity(
                        pred_conn.astype('uint8'), label_conn.astype('uint8'), pred_num_conn,
                        origin_pred_num_conn, label_num_conn, pred_i,
                        real_label_num, real_pred_num, zero)
                    sc_loss += 1 - img_connectivity
                elif real_label_num == 0 and real_pred_num == 0:
                    # if no connected component, SC Loss = 0, so pass
                    pass
                else:
                    preds_class = pred_i == int(class_)
                    not_preds_class = torch.bitwise_not(preds_class)
                    labels_class = torch.tensor(labels_np_class)
                    missed_detect = labels_class * not_preds_class
                    missed_detect_area = torch.sum(missed_detect).float()
                    sc_loss += missed_detect_area / missed_detect.numel() + 1

            multi_class_sc_loss[
                i] = sc_loss / class_num if class_num != 0 else 0
        multi_class_sc_loss = torch.mean(multi_class_sc_loss)
        return multi_class_sc_loss

    @property
    def loss_name(self):
        return self.loss_name_


def compute_class_connectiveity(pred_conn, label_conn, pred_num_conn,
                                origin_pred_num_conn, label_num_conn, pred,
                                real_label_num, real_pred_num, zero):
    pred_conn = torch.LongTensor(pred_conn)
    label_conn = torch.LongTensor(label_conn)
    pred_conn = F.one_hot(pred_conn, origin_pred_num_conn)
    label_conn = F.one_hot(label_conn, label_num_conn)

    ious = torch.zeros((real_label_num, real_pred_num))
    pair_conn_sum = torch.tensor([0.])

    for i in range(1, label_num_conn):
        label_i = label_conn[:, :, i]

        pair_conn = torch.tensor([0.])
        pair_conn_num = 0

        for j in range(1, pred_num_conn):
            pred_j_mask = pred_conn[:, :, j]
            pred_j = pred_j_mask * pred

            iou = compute_iou(pred_j, label_i, zero)
            ious[i - 1, j - 1] = iou
            if iou != 0:
                pair_conn += iou
                pair_conn_num += 1

        if pair_conn_num != 0:
            pair_conn_sum += pair_conn / pair_conn_num
    lone_pred_num = 0

    pred_sum = torch.sum(ious, dim=0)
    for m in range(0, real_pred_num):
        if pred_sum[m] == 0:
            lone_pred_num += 1
    img_connectivity = pair_conn_sum / (real_label_num + lone_pred_num)
    return img_connectivity


def compute_iou(pred_i, label_i, zero):
    intersect_area_i = torch.sum(pred_i * label_i)
    if torch.equal(intersect_area_i, zero):
        return 0

    pred_area_i = torch.sum(pred_i)
    label_area_i = torch.sum(label_i)
    union_area_i = pred_area_i + label_area_i - intersect_area_i
    if torch.equal(union_area_i, zero):
        return 1
    else:
        return intersect_area_i / union_area_i
