# -*- coding: utf-8 -*-
"""
@File    : refacter_pretrain.py
@Time    : 2020/4/21 10:07
@Author  : KeyForce
@Email   : july.master@outlook.com
"""
import argparse
import os

import torch


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_path", type=str, default='/root/PycharmProjects/mmdet/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth', help="the path of pretrained model")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    return parser.parse_args()


def modify_cascade_rcnn(model_coco, num_classes):
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"].resize_(num_classes, 1024)
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"].resize_(num_classes, 1024)
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"].resize_(num_classes, 1024)
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"].resize_(num_classes)
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"].resize_(num_classes)


def main(args):
    pth_dir = args.org_path
    model_coco = torch.load(pth_dir)
    # print(model_coco["state_dict"])

    modify_cascade_rcnn(model_coco, args.num_classes)
    # save new model
    model_name = '/root/PycharmProjects/mmdet/models/cascade_rcnn_x101_64x4d_fpn_1x_class_2.pth'
    torch.save(model_coco, model_name)
    print("Convert successful.")


if __name__ == '__main__':
    args = init_args()  #
    main(args)
