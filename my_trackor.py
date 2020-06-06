# -*- coding: utf-8 -*-
"""
@File    : my.py
@Time    : 2020/4/14 0:01
@Author  : KeyForce
@Email   : july.master@outlook.com
"""
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import cv2
import numpy as np
import torch
import json

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val
from tracktor import build_tracker

config_file = 'configs/MyDet/mot17_cascade_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file = '/root/PycharmProjects/mmdet/work_dirs/mot17_cascade_rcnn_x101_64x4d_fpn_1x/epoch_2.pth'
json_save_path = '/root/data/a.json'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

rootdir = '/root/data/A-data/Track9'
# rootdir = '/root/data/5'
outdir = '/root/data/TestALL/Track9'
filename = '/root/data/TestALL/Track9/Track9.txt'


def write_results(filename, results, frame1):
    save_format = '{frame},{id},{x1},{y1},{w},{h},0.9,0\n'

    frame1.extend(results)

    with open(filename, 'w') as f:
        for frame_id, xyxys, track_ids in frame1:
            x1, y1, x2, y2 = xyxys
            w = x2 - x1
            h = y2 - y1
            line = save_format.format(frame=frame_id, id=track_ids, x1=x1, y1=y1, w=w, h=h)
            f.write(line)

    f.close()
    return


def imshow(img, win_name='', wait_time=0):
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hangning if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0.3,
                      bbox_color='green',
                      text_color='white',
                      thickness=2,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      image_id=0):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    # results[image_id] = []

    # do tracking
    bbox_xyxy = bboxes[:,:4]
    cls_conf = bboxes[..., 4]
    # w = bbox_xyxy[..., 2] - bbox_xyxy[..., 0]
    # h = bbox_xyxy[..., 3] - bbox_xyxy[..., 1]
    # bbox_xyxy[:, 2] = w
    # bbox_xyxy[:, 3] = h

    index = 0
    outbb = np.copy(bbox_xyxy)
    dd = []
    for bbox, label in zip(bboxes, labels):
        if label != 0:
            dd.append(index)
        index += 1
    outbb = np.delete(outbb, dd, 0)
    cls_conf = np.delete(cls_conf, dd, 0)
    labels = np.delete(labels, dd, 0)
    # outbb [xywh]
    # outbb [xyxy]

    # outputs = deepsort.update(outbb, cls_conf, img)
    boxes = torch.from_numpy(outbb)
    cls_conf = torch.from_numpy(cls_conf)
    img_in = torch.from_numpy(img).unsqueeze(0).permute([0, 3, 1, 2])

    o_boxes, o_ids = track.step(boxes, cls_conf, img_in)

    if True:
        for bbox, ids in zip(o_boxes, o_ids):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = 'ID {}'.format(ids)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])

            _height_half = int((bbox_int[3] - bbox_int[1]) / 2)
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] + _height_half),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

            mybbox = bbox.astype(np.float).tolist()
            results.append((image_id, mybbox, ids))

            if image_id == 2:
                frame1.append((1, mybbox, ids))



        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)



if __name__ == '__main__':
    images = os.listdir(rootdir)
    images.sort(key=lambda x: int(x[:-4]))
    image_id = 1
    results = []
    frame1 = []
    track = build_tracker()

    for image in images:
        image_dir = os.path.join(rootdir, image)
        result = inference_detector(model, image_dir)
        print(image_dir)
        out_file = os.path.join(outdir, 'frame', image)
        # show_result(image_dir, result, model.CLASSES, out_file=out_dir)

        img = mmcv.imread(image_dir)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # draw bounding boxes
        imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=model.CLASSES,
            score_thr=0.3,
            show=False,
            wait_time=0,
            out_file=out_file,
            image_id=image_id)
        image_id +=1

    # json.dump(results, open(json_save_path, 'w'))



    if True:
        output_dir = os.path.join(outdir,'frame')
        output_video_path = os.path.join(outdir, '{}.mp4'.format('Track9'))
        cmd_str = 'ffmpeg -f image2 -i {}/%d.jpg -b 5000k -c:v mpeg4 {}'.format(output_dir, output_video_path)
        os.system(cmd_str)

    write_results(filename, results, frame1)





