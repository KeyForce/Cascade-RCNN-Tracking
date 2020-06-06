#这个脚本主要作用就是制作所谓的mot19数据，即通过测试数据进行合成数据
#先切出所用检测到的目标，在随机合成在第一帧的图片

import os
import random
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance


root_path = '/home/liucen/project/deepMOT/test_dataset/capture_pic/'                 #存储原始图片和坐标的位置
crop_pic_path = '/mnt/liucen/MOT_challenge/MOT_19/crop/'                             #需要保存的crop的图片位置

#切割图片===============================================================================================
# for ind in os.listdir(root_path):
#     pic_file = root_path + ind + '/img1/'
#     ann_file = root_path + ind + '/det/det.txt'
#
#     i = 0
#     for line in tqdm(open(ann_file)):
#         coord = line.strip('\n').split(',')
#         if float(coord[6]) <= 0.9:                                                 #设置切割的阈值
#             continue
#         elif (float(coord[2]) <= 5) or (float(float(coord[3])) <= 5):              #有在边缘的人像
#             continue
#         elif (ind in ['b2', 'b3']) and ((float(coord[2]) + float(coord[4]) >= 2555) or (float(coord[3]) + float(coord[5]) >= 1435)):
#             continue
#         elif (ind in ['b1', 'b4', 'b5']) and ((float(coord[2]) + float(coord[4]) >= 1915) or (float(coord[3]) + float(coord[5]) >= 1075)):
#             continue
#         else:
#             pic = pic_file + '{:>06d}.jpg'.format(int(coord[0]))                   #准备被切割的图片
#             im = Image.open(pic)
#             crop_pic = im.crop((float(coord[2]), float(coord[3]),
#                                 float(coord[2])+float(coord[4]),
#                                 float(coord[3])+float(coord[5])))
#             crop_pic.save(crop_pic_path + ind + '/' + str(i) + '.jpg')
#             i += 1

#合成图片=====================================================================================================
def Transform(pic):
    #对选中的行人图片进行数据增强，同时返回行人图片的高和宽
    #读取图片
    image = Image.open(pic)

    #改变亮度
    brightness = random.uniform(0.8, 1.2)
    enh_bri = ImageEnhance.Brightness(image)
    image = enh_bri.enhance(brightness)

    #改变色度
    color = random.uniform(0.8, 1.2)
    enh_col = ImageEnhance.Color(image)
    image = enh_col.enhance(color)

    #改变对比度
    contrast = random.uniform(0.8, 1.2)
    enh_con = ImageEnhance.Contrast(image)
    image = enh_con.enhance(contrast)

    #改变锐度
    shapeness = random.uniform(0.8, 1.2)
    enh_sha = ImageEnhance.Sharpness(image)
    image = enh_sha.enhance(shapeness)

    return image, image.size[0], image.size[1]

def Compound(bg, ind, clean=True):
    #合成图片， 传进来参数是背景， 序号， 是否是干净的背景
    prople_totle = len(os.listdir(crop_pic_path + ind))-1                           #可选行人总数
    if clean:
        ann_list = []
        people_num = random.randint(10, 20)                                       #图片中添加的行人个数
    else:
        if ind == 'b4':
            ann_list = [[692,65,55,151],[715,0,42,78],[696,232,68,222],[1109,293,78,221],[939,17,55,139]]
            people_num = random.randint(5, 10)
        else:
            ann_list = [[260,328,99,203]]
            people_num = random.randint(10, 40)
    bg_pic = Image.open(bg)
    bg_w, bg_h = bg_pic.size
    for n in range(people_num):
        peo_id = random.randint(0, prople_totle)
        peo_pic = crop_pic_path + ind + '/' + str(peo_id) + '.jpg'

        peo_pic_t, peo_w, peo_h = Transform(peo_pic)

        x = random.randint(0, bg_w-peo_w-1)
        y = random.randint(-peo_h, bg_h)

        #对people是否在边缘进行讨论
        if y <= 0:
            #当行人一半被上半边切开
            h = peo_h + y,
            h = h[0]
            w = peo_w
            peo_pic_t = peo_pic_t.crop((0, -y, peo_w, peo_h))
            yy = 0
            xx = x
        elif y + peo_h > bg_h:
            #当行人一半被下半边切开
            h = bg_h - y
            w = peo_w
            peo_pic_t = peo_pic_t.crop((0, 0, w, h))
            yy = y
            xx = x
        else:
            #正常
            h = peo_h
            w = peo_w
            xx = x
            yy = y

        bg_pic.paste(peo_pic_t, (xx, yy))
        # bg_pic.show()
        ann_list.append([xx, yy, w, h])
    return bg_pic, ann_list

ind_list = ['b2', 'b3', 'b4', 'b5']
dir_back = {'b2': '000090.jpg','b3':'000100.jpg', 'b4': '000004.jpg', 'b5': '000001.jpg'}                                                                    #背景对应的图片

pic_id = 1
f = open('/mnt/liucen/MOT_challenge/MOT_19/det/det.txt', 'a')
for ind in ind_list:
    if ind == 'b2' or ind == 'b3':                                                #原图背景干净
        for i in tqdm(range(1000)):                                               #生成1000张
            background = root_path + ind + '/img1/' + dir_back[ind]
            pic, ann_list = Compound(background, ind, clean=True)
            #保存图片与标签
            pic.save('/mnt/liucen/MOT_challenge/MOT_19/img1/' + '{:>06d}.jpg'.format(int(pic_id)))
            for ann in ann_list:
                f.write(str(pic_id)+','+'-1,')
                for i in ann:
                    f.write(str(i)+',')
                f.write('1,1,1\n')
            pic_id+=1
    else:                                                                      #原图背景不干净
        for i in tqdm(range(1000)):                                            #生成1000张
            background = root_path + ind + '/img1/' + dir_back[ind]
            pic, ann_list = Compound(background, ind, clean=True)
            #保存图片与标签
            pic.save('/mnt/liucen/MOT_challenge/MOT_19/img1/' + '{:>06d}.jpg'.format(int(pic_id)))
            for ann in ann_list:
                f.write(str(pic_id)+','+'-1,')
                for i in ann:
                    f.write(str(i)+',')
                f.write('1,1,1\n')
            pic_id+=1