# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
from tqdm import tqdm
import numpy as np
import json
import scipy.io as sio                       #读写m文件要用
from PIL import Image

root_path = '/mnt/liucen/PRW-v16.04.20/'
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {'1': 1}

#把每个m文件进行处理，返回一个字典，字典的key就是帧，值是个list，其中包括id，坐标，类别
def m2dir(label_txt):
    f = open(os.path.join(root_path, label_txt), 'r')
    dir = {}

    l = []
    #初始化字典
    for line in f:
        frame = line.split(',')[0]
        l.append(frame)
    for i in set(l):
        dir[i] = []
    f.close()

    f = open(os.path.join(root_path, label_txt), 'r')
    #补充字典
    for line in f:
        [frame, id, x, y, w, h, conf, cag, _] = line.split(',')
        if conf == '0' or cag in ['3','4','5','6','8','9','10','11','12']:
            continue
        else:
            dir[frame].append([x, y, w, h, cag])
    return dir

#返回图片的长宽，输入的是图片的名字和frame
def get_size(filename):
    pic = root_path + 'frames/' + filename
    im = Image.open(pic)
    return pic, im.size[0], im.size[1]

#传进来是标签列表名字，标签根目录，要保存的文件名字
def convert(label_list, json_file):
    '''
    :param label_list: 需要转换的m文件列表
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
    for label_m in tqdm(label_list):                                        #循环标签列表
        filename = label_m.split('.')[0] + '.jpg'
        # 解析m文件
        m = sio.loadmat(root_path + 'annotations/' + label_m)

        #由于标注文件的检测框字段可能是‘box_new’,也可能是‘anno_file’,因次这块需要一个判断
        if 'box_new' in m:
            m_ann = m['box_new']
        if 'anno_file' in m:
            m_ann = m['anno_file']

        ## The filename must be a number
        image_id = imageID  # 图片ID
        imageID += 1

        # 图片的基本信息
        picfile, width, height = get_size(filename)
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)

        # 复制图片
        shutil.copyfile(picfile, '/mnt/liucen/MOT_det/coco_prw/train2017/' + filename)

        for i in range(m_ann.shape[0]):
            obj = m_ann[i]       #从第一行开始取坐标
            category_id = 1

            #设置坐标
            xmin = obj[1]
            ymin = obj[2]
            w = obj[3]
            h = obj[4]

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

def PRW_TO_COCO():
    #设置转换的label文件
    label_list = os.listdir(root_path + 'annotations/')           #PRW中存放标注文件的文件夹
    json_file = '/mnt/liucen/MOT_det/coco_prw/annotations/instances_train2017.json'
    convert(label_list, json_file)                                  #转换

if __name__ == '__main__':
    PRW_TO_COCO()