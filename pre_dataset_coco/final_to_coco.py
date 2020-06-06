#------------------------------------
#当把Wider和MOT数据集转换好coco格式之后
#合并两个数据集
#-----------------------------------

# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import shutil
from tqdm import tqdm
import numpy as np
import json
from PIL import Image

from pre_dataset_coco.mot17_to_coco import MOT17_TO_COCO
from pre_dataset_coco.mot15_to_coco import MOT15_TO_COCO
from pre_dataset_coco.wider_to_cooc import WIDER_TO_COCO
from pre_dataset_coco.PRW_to_coco import PRW_TO_COCO

root_path = '/home/omnisky/Bob/coco/'
# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {'1': 1}

#返回图片的长宽，输入的是json名字和frame
def get_size(json_file, frame):
    p = json_file.split('/')[5]
    if p == 'coco_person':
        pic = root_path + 'coco_person/train2017/' + frame
    elif p == 'coco_mot16':
        pic = root_path + 'coco_mot16/train2017/' + frame
    elif p == 'coco_mot17':
        pic = root_path + 'coco_mot17/train2017/' + frame
    elif p == 'coco_prw':
        pic = root_path + 'coco_prw/train2017/' + frame
    elif p == 'coco_mot19':
        pic = root_path + 'coco_mot19/train2017/' + frame
    im = Image.open(pic)
    return pic, im.size[0], im.size[1]

#解析json文件，返回一个字典，字典的key是图片名字，值是坐标的列表
def analysis_json(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

        dir = {}
        id_to_iamge = {}                                               #制作一个字典，为了图片名字和ID对应
        for pic_name in json_dict['images']:
            dir[pic_name['file_name']] = []
            id_to_iamge[pic_name['id']] = pic_name['file_name']

        for bbox in json_dict['annotations']:
            pic_name = id_to_iamge[bbox['image_id']]
            dir[pic_name].append(bbox['bbox'])

        return dir

#传进来是json标签列表名字，标签根目录，要保存的文件名字
def convert(json_list, json_file):
    '''
    :param json_list: 需要转换的txt列表
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
    for json_f in json_list:                                        #循环标签列表

        #解析json文件
        json_dir = analysis_json(json_f)

        for frame in tqdm(json_dir.keys()):
            # filename = frame


            ## The filename must be a number
            image_id = imageID                # 图片ID
            imageID += 1
            filename = '{:>06d}.jpg'.format(int(image_id))

            # 图片的基本信息
            picfile, width , height = get_size(json_f, frame)
            image = {'file_name': filename,
                     'height': height,
                     'width': width,
                     'id':image_id}
            json_dict['images'].append(image)

            #复制图片
            shutil.copyfile(picfile, '/home/omnisky/Bob/coco/coco_mot_all/train2017/'+filename)

            ## Cruuently we do not support segmentation
            #  segmented = get_and_check(root, 'segmented', 1).text
            #  assert segmented == '0'
            # 处理每个标注的检测框
            for obj in json_dir[frame]:
                #设置类别
                category_id = 1

                #设置坐标
                xmin = int(obj[0])
                ymin = int(obj[1])
                w = int(obj[2])
                h = int(obj[3])

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

    cat = {'supercategory': 'none', 'id': 1, 'name': 'person'}
    json_dict['categories'].append(cat)
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

if __name__ == '__main__':
    # #MOT ——> Coco
    # print("MOT17 数据集开始转换.....")
    # MOT17_TO_COCO()
    # print("MOT 数据集转换完成!")
    #
    # # MOT ——> Coco
    # print("MOT15 数据集开始转换.....")
    # MOT15_TO_COCO()
    # print("MOT 数据集转换完成!")
    #
    # #PRW ————> Coco
    # print("PRW数据集开始转换.....")
    # PRW_TO_COCO()
    # print("PRW 数据集转换完成!")

    #Wider ——> Coco
    # print("WIDER 数据集开始转换.....")
    # WIDER_TO_COCO()
    # print("WIDER 数据集转换完成!")

    #复制验证集数据
    # print("开始复制验证集.....")
    # shutil.copytree("/mnt/liucen/MOT_det/coco_wider/val2017",
    #                 "/mnt/liucen/MOT_det/coco/val2017")
    # shutil.copyfile("/mnt/liucen/MOT_det/coco_wider/annotations/instances_val2017.json",
    #                 "/mnt/liucen/MOT_det/coco/annotations/instances_val2017.json")
    # print("验证集复制完成!")

    #设置转换的label文件
    print("开始最后的合并.....")
    # json_list = ["/home/omnisky/Bob/coco/coco_person/instances_coco_person_train2017.json",
    #              "/home/omnisky/Bob/coco/coco_mot17/instances_mot17_train2017.json",
    #              "/home/omnisky/Bob/coco/coco_mot16/instances_mot16_train2017.json"]
    json_list = ["/home/omnisky/Bob/coco/coco_mot17/instances_mot17_train2017.json",
                 "/home/omnisky/Bob/coco/coco_mot16/instances_mot16_train2017.json"]
    json_file = "/home/omnisky/Bob/coco/coco_mot_all/instances_train2017.json"
    convert(json_list, json_file)
    print("检测数据集制作完成!")
