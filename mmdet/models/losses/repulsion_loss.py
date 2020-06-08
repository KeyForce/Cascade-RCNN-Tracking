# -*- coding: utf-8 -*-
"""
@File    : repulsion_loss.py
@Time    : 2020/6/8 21:24
@Author  : KeyForce
@Email   : july.master@outlook.com
"""
import math
import torch
from torch.autograd import Variable
import numpy as np
import pdb

def bbox_transform_inv(boxes, deltas, batch_size):

    #import pdb
    #pdb.set_trace()
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0]
    dy = deltas[:, :, 1]
    dw = deltas[:, :, 2]
    dh = deltas[:, :, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros(boxes.shape[0],boxes.shape[1],4).float().cuda()
    # x1
    pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * pred_h

    #print(pred_ctr_x-0.5*pred_w)
    #import pdb
    #pdb.set_trace()

    return pred_boxes

def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)
    #import pdb
    #pdb.set_trace()
    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def IoG(box_a, box_b):
    inter_xmin = torch.max(box_a[0], box_b[0])
    inter_ymin = torch.max(box_a[1], box_b[1])
    inter_xmax = torch.min(box_a[2], box_b[2])
    inter_ymax = torch.min(box_a[3], box_b[3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return I / G


def repgt(pred_boxes, gt_rois, rois_inside_ws):
    sigma_repgt = 0.9
    loss_repgt = torch.zeros(pred_boxes.shape[0]).cuda()
    for i in range(pred_boxes.shape[0]):
        boxes = Variable(
            pred_boxes[i, rois_inside_ws[i] != 0].view(int(pred_boxes[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))
        gt = Variable(gt_rois[i, rois_inside_ws[i] != 0].view(int(gt_rois[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))
        num_repgt = 0
        repgt_smoothln = 0
        if boxes.shape[0] > 0:
            overlaps = bbox_overlaps(boxes, gt)
            for j in range(overlaps.shape[0]):
                for z in range(overlaps.shape[1]):
                    if int(torch.sum(gt[j] == gt[z])) == 4:
                        overlaps[j, z] = 0
            max_overlaps, argmax_overlaps = torch.max(overlaps, 1)
            for j in range(max_overlaps.shape[0]):
                if max_overlaps[j] > 0:
                    num_repgt += 1
                    iog = IoG(boxes[j], gt[argmax_overlaps[j]])
                    if iog > sigma_repgt:
                        repgt_smoothln += ((iog - sigma_repgt) / (1 - sigma_repgt) - math.log(1 - sigma_repgt))
                    elif iog <= sigma_repgt:
                        repgt_smoothln += -math.log(1 - iog)
        if num_repgt > 0:
            loss_repgt[i] = repgt_smoothln / num_repgt

    return loss_repgt


def repbox(pred_boxes, gt_rois, rois_inside_ws):
    sigma_repbox = 0
    loss_repbox = torch.zeros(pred_boxes.shape[0]).cuda()

    for i in range(pred_boxes.shape[0]):

        boxes = Variable(
            pred_boxes[i, rois_inside_ws[i] != 0].view(int(pred_boxes[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))
        gt = Variable(gt_rois[i, rois_inside_ws[i] != 0].view(int(gt_rois[i, rois_inside_ws[i] != 0].shape[0]) / 4, 4))

        num_repbox = 0
        repbox_smoothln = 0
        if boxes.shape[0] > 0:
            overlaps = bbox_overlaps(boxes, boxes)
            for j in range(overlaps.shape[0]):
                for z in range(overlaps.shape[1]):
                    if z >= j:
                        overlaps[j, z] = 0
                    elif int(torch.sum(gt[j] == gt[z])) == 4:
                        overlaps[j, z] = 0

            iou = overlaps[overlaps > 0]
            for j in range(iou.shape[0]):
                num_repbox += 1
                if iou[j] <= sigma_repbox:
                    repbox_smoothln += -math.log(1 - iou[j])
                elif iou[j] > sigma_repbox:
                    repbox_smoothln += ((iou[j] - sigma_repbox) / (1 - sigma_repbox) - math.log(1 - sigma_repbox))

        if num_repbox > 0:
            loss_repbox[i] = repbox_smoothln / num_repbox

    return loss_repbox


def repulsion(rois, box_deltas, gt_rois, rois_inside_ws, rois_outside_ws):
    deltas = Variable(box_deltas.view(rois.shape[0], 256, 4))
    rois_inside_ws = Variable(rois_inside_ws.view(rois.shape[0], 256, 4))
    rois_outside_ws = Variable(rois_outside_ws.view(rois.shape[0], 256, 4))
    if int(torch.sum(rois_outside_ws == rois_inside_ws)) != 1024:
        import pdb
        pdb.set_trace()

    BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

    for i in range(rois.shape[0]):
        deltas[i] = deltas[i].view(-1, 4) * torch.FloatTensor(BBOX_NORMALIZE_STDS).cuda() + \
                    torch.FloatTensor(BBOX_NORMALIZE_MEANS).cuda()

    pred_boxes = bbox_transform_inv(rois[:, :, 1:5], deltas, 2)

    loss_repgt = repgt(pred_boxes, gt_rois, rois_inside_ws)
    loss_repbox = repbox(pred_boxes, gt_rois, rois_inside_ws)

    return loss_repgt, loss_repbox