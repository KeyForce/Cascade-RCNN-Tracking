# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
from tqdm import tqdm
import numpy as np
import json
from PIL import Image

root_path = '/mnt/liucen/MOT_challenge/2DMOT2015/train/'
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {'1': 1}

#把每个txt文件进行处理，返回一个字典，字典的key就是帧，值是个list，其中包括id，坐标，类别
def txt2dir(label_txt):
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
        [frame, id, x, y, w, h, conf, cag, _, _] = line.split(',')
        if conf == '0':
            continue
        else:
            dir[frame].append([x, y, w, h, cag])
    return dir

#返回图片的长宽，输入的是label_txt名字和frame
def get_size(label_txt, frame):
    p = label_txt.split('/')[0] + '/' + 'img1/{:>06d}.jpg'.format(int(frame))
    pic = root_path + p
    im = Image.open(pic)
    return pic, im.size[0], im.size[1]

#传进来是标签列表名字，标签根目录，要保存的文件名字
def convert(label_list, json_file):
    '''
    :param label_list: 需要转换的txt列表
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
    for label_txt in label_list:                                        #循环标签列表
        video_id = label_txt.split('/')[0]
        print("buddy~ Processing {}".format(video_id))
        # 解析TXT
        txt_dir = txt2dir(label_txt)

        for frame in tqdm(txt_dir.keys()):
            filename = video_id + '_{:>06d}.jpg'.format(int(frame))

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

            #复制图片
            shutil.copyfile(picfile, '/mnt/liucen/MOT_det/coco_mot15/train2017/'+filename)

            ## Cruuently we do not support segmentation
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            # 处理每个标注的检测框
            for obj in txt_dir[frame]:
                #设置类别
                # category = obj[-1]
                # if category not in categories:
                #     new_id = len(categories)
                #     categories[category] = new_id
                # category_id = categories[category]
                category_id = 1

                #设置坐标
                xmin = int(float(obj[0]))
                ymin = int(float(obj[1]))
                w = int(float(obj[2]))
                h = int(float(obj[3]))

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

def MOT15_TO_COCO():
    #设置转换的label文件
    label_list = ['Venice-2/gt/gt.txt','TUD-Stadtmitte/gt/gt.txt', 'TUD-Campus/gt/gt.txt',
                  'PETS09-S2L1/gt/gt.txt', 'KITTI-17/gt/gt.txt', 'KITTI-13/gt/gt.txt',
                  'ETH-Sunnyday/gt/gt.txt', 'ETH-Pedcross2/gt/gt.txt', 'ETH-Bahnhof/gt/gt.txt',
                  'ADL-Rundle-8/gt/gt.txt', 'ADL-Rundle-6/gt/gt.txt']
    json_file = '/mnt/liucen/MOT_det/coco_mot15/annotations/instances_train2017.json'
    convert(label_list, json_file)                                  #转换

if __name__ == '__main__':
    MOT15_TO_COCO()