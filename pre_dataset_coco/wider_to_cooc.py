#-------------------------------------------------------
#处理wider数据集，制作验证集和训练集
#只用wider中的sur开头（监控来源）的数据集，用wider中的验证集来验证
#-------------------------------------------------------

# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
from tqdm import tqdm
import numpy as np
import json
from PIL import Image

root_path = '/mnt/liucen/Wider/'
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {'1': 1}

#处理标签的txt文件，返回一个字典，字典的key就是图片名字，值是个list，其中包括坐标
#注意指出里sur开头的图片，不处理ad开头的
def txt2dir(label_txt):
    f = open(os.path.join(root_path, label_txt), 'r')
    dir = {}

    l = []
    #初始化字典
    for line in f:
        line = line.strip('\n')                   #去除文本的结尾换行符
        inf_list = line.split(' ')
        if inf_list[0][0] == 'a':                 #去除‘ad’的图像
            continue
        else:
            pic_name = inf_list[0]
            dir[pic_name] = []
    f.close()

    f = open(os.path.join(root_path, label_txt), 'r')
    #补充字典
    for line in f:
        line = line.strip('\n')
        inf_list = line.split(' ')
        pic_name = inf_list[0]
        del inf_list[0]                         #删除列表中第一个元素（图片名字）

        if pic_name[0] == 'a':
            continue
        else:
            num_box = int(len(inf_list) / 4)        #每行有多少个bbox
            for i in range(num_box):
                st = i * 4                       #每个bbox其实位置的下标
                bbox = [int(inf_list[st]), int(inf_list[st+1]), int(inf_list[st+2]), int(inf_list[st+3])]
                dir[pic_name].append(bbox)
    f.close()
    return dir

#返回图片的长宽，输入的是label_txt名字和frame
def get_size(label_txt, frame):
    p = label_txt.split('/')[1].split('_')[0]
    if p == 'val':
        pic = root_path + 'val_data/' + frame
    else:
        pic = root_path + 'sur_train/' + frame
    im = Image.open(pic)
    return pic, im.size[0], im.size[1]

#传进来是标签文件名字，标签根目录，要保存的文件名字
def convert(label_file, json_file):
    '''
    :param label_file: 需要转换的txt文件
    :param json_file: 导出json文件的路径
    :return: None
    '''

    # 标注基本结构
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    imageID = 1
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    label_txt = label_file                       #循环标签列表

    # video_id = label_txt.split('/')[0].split('-')[1]
    # print("buddy~ Processing {}".format(video_id))
    # 解析TXT
    txt_dir = txt2dir(label_txt)

    for frame in tqdm(txt_dir.keys()):
        filename = frame                  #得到图片名字

        ## The filename must be a number
        image_id = imageID                # 图片ID
        imageID += 1

        # 图片的基本信息
        picfile, width , height = get_size(label_txt, frame)
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)

        #复制图片到一个制定路径
        if picfile.split('/')[-2].split('_')[0] == 'sur':
            shutil.copyfile(picfile, '/mnt/liucen/MOT_det/coco_wider/train2017/'+filename)
        else:
            shutil.copyfile(picfile, '/mnt/liucen/MOT_det/coco_wider/val2017/'+filename)

        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        # 处理每个标注的检测框
        for bbox in txt_dir[frame]:
            #设置类别
            # category = obj[-1]
            # if category not in categories:
            #     new_id = len(categories)
            #     categories[category] = new_id
            category_id = 1

            #设置坐标
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])

            annotation = dict()
            annotation['area'] = w * h
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, w, h]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin, ymin, xmin, ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

def WIDER_TO_COCO():
    #制作验证集
    # 设置转换的label文件
    label_file = 'Annotations/val_bbox.txt'
    json_file = '/mnt/liucen/MOT_det/coco_wider/annotations/instances_val2017.json'
    convert(label_file, json_file)

    #制作训练集
    #设置转换的label文件
    label_file = 'Annotations/train_bbox.txt'
    json_file = '/mnt/liucen/MOT_det/coco_wider/annotations/instances_train2017.json'
    convert(label_file, json_file)                                  #转换


