from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorboardX import SummaryWriter
import os
from dataload import listDataset
import torch
import torch.utils.data
from opts import opts
import logging
import time
import torchvision
from torchvision import datasets, transforms
import numpy as np
# from losses import CtdetLoss
import cv2
import torch.nn as nn
from eval_utils import load_model,pre_process,process,post_process,merge_outputs,add_coco_bbox


coco_class_name = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255


def detect_eval(image,model,num_classes,max_per_image):
    images,meta = pre_process(image,1)
    images = images.to("cuda")
    output,dets= process(images,model,return_time=True)
    detection_result = []
    dets = post_process(dets,meta,num_classes)
    results = merge_outputs(dets,num_classes,max_per_image)
    # images = images.to("cpu")
    for j in range(1, num_classes + 1):
        for bbox in results[j]:
          if bbox[4] > 0.001:
            detection_result.append([bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],j-1])
    
    return detection_result


def detect(image,model,num_classes):
    images,meta = pre_process(image,1)
    
    images = images.to("cuda")
    output,dets= process(images,model,return_time=True)
    # print("the wh is {}".format(output["wh"]))
    dets = post_process(dets,meta,num_classes)
    results = merge_outputs(dets,num_classes,max_per_image)

    for j in range(1, num_classes + 1):
        for bbox in results[j]:
          if bbox[4] > 0.3:
              image_detection = add_coco_bbox(image,bbox, bbox[4], conf=1, show_txt=True, img_id='default')
    cv2.imshow("detection",image_detection)
    cv2.waitKey(1)
    return image_detection


if __name__ == '__main__':
    # image = cv2.imread("./54.jpg")
    num_classes = 20
    max_per_image = 100
    from models import get_pose_net
    heads = {"hm":num_classes,"wh":2,"reg":2}
    model = get_pose_net(50,heads, head_conv=64)    
    # model = load_model(model,"ctdet_coco_dla_2x.pth")
    model = load_model(model,"resnet50dcn.pth")
    model.cuda()
    model.eval()
    video = cv2.VideoCapture("t640480_det_results.avi")
    
    while True:
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
    
        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        image_result = detect(frame,model,num_classes)
        # cv2.imshow("test",image_result)
        # cv2.waitKey(10)
