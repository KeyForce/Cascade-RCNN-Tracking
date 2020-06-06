# -*- coding: utf-8 -*-
"""
@File    : coco2person.py
@Time    : 2020/4/25 16:37
@Author  : KeyForce
@Email   : july.master@outlook.com
"""
from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json


def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def showimg(coco, img, classes, cls_id, show=True):
    global dataDir
    # I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    # plt.show()
    objs = []
    I = Image.open(os.path.join('/home/omnisky/Bob/coco/coco_all/train2017', img['file_name']))
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            # print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)

                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs




if __name__ == '__main__':
    annFile = '/home/omnisky/Bob/coco/coco_person/instances_coco_person_train2017.json'
    coco = COCO(annFile)
    classes = id2name(coco)
    print(classes)
    classes_names = ['person']
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)

    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    imageID = 1
    bnd_id = 1

    for cls in classes_names:
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)

        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            # print(filename)
            objs = showimg(coco, img, classes, classes_ids, show=True)
            # print(objs)

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=classes_ids, iscrowd=None)
            # print(annIds)
            anns = coco.loadAnns(annIds)

            image_id = imageID
            imageID += 1

            image = {'file_name': img['file_name'],
                     'height': img['height'],
                     'width': img['width'],
                     'id': image_id}
            json_dict['images'].append(image)

            for ann in anns:
                annotation = dict()
                annotation['area'] = ann['area']
                annotation['iscrowd'] = 0
                annotation['image_id'] = image_id
                annotation['bbox'] = ann['bbox']
                annotation['category_id'] = 1
                annotation['id'] = bnd_id
                annotation['ignore'] = 0
                # 设置分割数据，点的顺序为逆时针方向
                # annotation['segmentation'] = ann['segmentation']
                annotation['segmentation'] = []

                json_dict['annotations'].append(annotation)
                bnd_id = bnd_id + 1

    cat = {'supercategory': 'person', 'id': 1, 'name': 'person'}
    json_dict['categories'].append(cat)

    json_file = '/root/data/coco/instances_coco_person_train2017.json'
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()






