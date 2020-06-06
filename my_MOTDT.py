# -*- coding: utf-8 -*-
"""
@File    : my_motdt.py
@Time    : 2020/5/22 10:27
@Author  : KeyForce
@Email   : july.master@outlook.com
"""
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import cv2
import numpy as np
import json
import shutil
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val
from deep_sort import build_tracker
from MOTDT.tracker.mot_tracker import OnlineTracker
# from mgn_sort import build_tracker
import argparse


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},0.9,0\n'

    with open(filename, 'w') as f:
        for frame_id, xyxys, track_ids in results:
            x1, y1, x2, y2 = xyxys
            w = x2 - x1
            h = y2 - y1
            line = save_format.format(frame=frame_id, id=track_ids, x1=x1, y1=y1, w=w, h=h)
            f.write(line)

    f.close()
    return

def new_write_results(filename, results):
    save_format = '{frame},-1,{x1},{y1},{w},{h},{conf}\n'

    with open(filename, 'w') as f:
        for frame_id, xyxys, track_ids, conf in results:
            x1, y1, x2, y2 = xyxys
            w = x2 - x1
            h = y2 - y1
            line = save_format.format(frame=frame_id, id=track_ids, x1=x1, y1=y1, w=w, h=h, conf=conf)
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
    w = bbox_xyxy[..., 2] - bbox_xyxy[..., 0]
    h = bbox_xyxy[..., 3] - bbox_xyxy[..., 1]
    bbox_xyxy[:, 2] = w
    bbox_xyxy[:, 3] = h


    # 去除label不是人的BBox
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

    # 去除BBox过小
    BBox_area_sixze = w * h
    index_filter = 0
    outbb_filter = np.copy(outbb)
    dd_filter = []
    for bbox, size in zip(outbb_filter, BBox_area_sixze):
        if size <= 200:
            dd_filter.append(index_filter)
        index_filter += 1
    outbb_filter = np.delete(outbb_filter, dd_filter, 0)
    cls_conf = np.delete(cls_conf, dd_filter, 0)
    labels = np.delete(labels, dd_filter, 0)

    # 去除BBox过小
    BBox_area_h = h
    index_filter = 0
    ddd_filter = []
    for bbox, size in zip(outbb_filter, BBox_area_h):
        if size <= 35:
            ddd_filter.append(index_filter)
        index_filter += 1
    outbb_filter = np.delete(outbb_filter, ddd_filter, 0)
    cls_conf = np.delete(cls_conf, ddd_filter, 0)
    labels = np.delete(labels, ddd_filter, 0)



    # if image_id ==1:
    #     for i in range(4):
    #         outputs = tracker.update(img, outbb_filter, cls_conf)
    # else:
    #     outputs = tracker.update(img, outbb_filter, cls_conf)
    outputs = tracker.update(img, outbb_filter, cls_conf)

    online_tlwhs = []
    online_ids = []
    for t in outputs:
        online_tlwhs.append(t.tlwh)
        online_ids.append(t.track_id)


    if len(online_ids) > 0:
        bboxes = online_tlwhs
        ids = online_ids

        for bbox, label, id, conf in zip(bboxes, labels, ids, cls_conf):
            if label == 0:
                bbox_int = bbox.astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[0]+bbox_int[2], bbox_int[1]+bbox_int[3])
                cv2.rectangle(
                    img, left_top, right_bottom, bbox_color, thickness=thickness)
                label_text = 'ID{} cnf{:.2}'.format(id, conf)
                if len(bbox) > 4:
                    label_text += '|{:.02f}'.format(bbox[-1])

                _height_half = int((bbox_int[3]-bbox_int[1])/2)
                cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] + _height_half),
                            cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

                mybbox = bbox.astype(np.float).tolist()
                results.append((image_id, mybbox, id))
                # results.append((image_id, mybbox, id, conf))


        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', default='configs/MyDet/coco_mot_cascade_rcnn_dconv_c3-c5_fpn_1x.py', help='test config file path')
    parser.add_argument('--checkpoint', default='/home/wild/Fucu/work_dirs/coco_crowdhuman_resxt_101_focalloss_giou_cascade_rcnn_dconv_c3-c5_fpn_1x/latest.pth', help='checkpoint file')
    parser.add_argument('--exp_name', default='8_FilterBBox_Cascade_Mask_RCNN_CrowdHuman_HRNet_epoch_34', type=str, help='the dir of input images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = parse_args()

    config_file = arg.config
    checkpoint_file = arg.checkpoint
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    exp_name = arg.exp_name
    # track_numbers = [1, 4, 5, 9, 10]
    track_numbers = [1]

    for track_number in track_numbers:
        rootdir = '/home/wild/Fucu/work/A-data/Track{0}'.format(track_number)
        # rootdir = '/root/data/5'
        outdir = '/home/wild/Fucu/work/{1}/Track{0}'.format(track_number, exp_name)
        filename = '/home/wild/Fucu/work/{1}/Track{0}/Track{0}.txt'.format(track_number, exp_name)

        images = os.listdir(rootdir)
        images.sort(key=lambda x: int(x[:-4]))
        image_id = 1

        results = []

        # deepsort = build_tracker('mgn_sort/deep/checkpoint/model.pt', max_dist=0.3, min_confidences=0.3,
        #                          nms_max_overlap=0.8, max_iou_distance=0.8,
        #                          max_age=80, n_init=3, nn_budget=200,
        #                          use_cuda=True)
        tracker = OnlineTracker()


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
                score_thr=0.2,
                show=False,
                wait_time=0,
                out_file=out_file,
                image_id=image_id)
            image_id += 1

        # json.dump(results, open(json_save_path, 'w'))

        if True:
            output_dir = os.path.join(outdir, 'frame')
            output_video_path = os.path.join(outdir, 'Track{0}.mp4'.format(track_number))
            cmd_str = 'ffmpeg -f image2 -i {}/%d.jpg -b 5000k -c:v mpeg4 {}'.format(output_dir, output_video_path)
            os.system(cmd_str)

        # shutil.rmtree(output_dir)

        # new_write_results(filename, results)
        write_results(filename, results)







