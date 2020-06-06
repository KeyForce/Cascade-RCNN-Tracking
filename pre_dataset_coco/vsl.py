import os
import sys
import cv2
import numpy as np

from pycocotools.coco import COCO

from skimage import io
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
dataset_dir = "/home/liucen/project/JD_tiger/train_data/det_train/coco/"      #更改
subset = "train"
year = "2017"

coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['1'])
imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds(imgIds=[1000])                                                       #更改
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]


img1 = cv2.imread('%s/%s'%('/home/liucen/project/JD_tiger/train_data/det_train/coco/OLD/train2017/',img['file_name']))    #更改

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
for i in anns:
    xy = (i['bbox'])
    cv2.rectangle(img1, (int(xy[0]), int(xy[1])), (int(xy[0] + xy[2]), int(xy[1] + xy[3])), (255, 255, 255), thickness=2)
cv2.imshow('head', img1)
cv2.waitKey(0)

