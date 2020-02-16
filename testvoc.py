from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# from dataload import listDataset
import torch
import torch.utils.data
from opts import opts
import logging
import time
import torchvision
from torchvision import datasets, transforms
import numpy as np
import cv2
import torch.nn as nn
from eval_utils import load_model,_nms,_tranpose_and_gather_feat,merge_outputs


VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
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
            0.667, 0.667, 0.000
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

  
def pre_process(image,scale, meta=None):
    height, width = image.shape[0:2]
    mean = np.array([[[0.408,0.447,0.47 ]]])
    std = np.array([[[0.289, 0.274,0.278]]])

    new_width,new_height = 384,384

    inp_image = cv2.resize(image,(new_width,new_height))
    
    inp_image = (inp_image.astype(np.float32) / 255.0).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, new_width, new_height)
    images = images.astype(np.float32)
    images = torch.from_numpy(images)

    meta = {'c': new_width/width, 's': new_height/height,
            'out_height': new_width // 4,
            'out_width': new_height // 4}
    
    return images,meta


def process(images, model,return_time=False):
    with torch.no_grad():
        output = model(images)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        torch.cuda.synchronize()
        dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=100)

    return output, dets


def post_process(dets,meta,num_classes,max_per_image,scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(),meta,num_classes,max_per_image)
    for j in range(1, num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()
    # print(batch, cat, height, width)
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs*4 - wh[..., 0:1] / 2*4, 
                        ys*4 - wh[..., 1:2] / 2*4,
                        xs*4 + wh[..., 0:1] / 2*4, 
                        ys*4 + wh[..., 1:2] / 2*4], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections


def add_coco_bbox(imgs, bbox, cat, conf=1, show_txt=True, img_id='default'): 
    bbox = np.array(bbox, dtype=np.int32)
    cat = int(cat)
    c = color_list[cat].tolist()
    txt = '{}{:.1f}'.format(VOC_CLASSES[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
      imgs, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
      cv2.rectangle(imgs,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(imgs, txt, (bbox[0], bbox[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return imgs


def detect_eval(image,model,num_classes,max_per_image):
    images,meta = pre_process(image,1)
    images = images.to("cuda")
    output,dets= process(images,model,return_time=True)
    detection_result = []
    dets = post_process(dets,meta,num_classes,max_per_image)
    results = merge_outputs(dets,num_classes,max_per_image)
    # images = images.to("cpu")
    for j in range(1, num_classes + 1):
        for bbox in results[j]:
          if bbox[4] > 0.1:
            detection_result.append([bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],j-1])
    
    return detection_result    


def detect(image,model,num_classes,max_per_image):
    images,meta = pre_process(image,1)
    images = images.to("cuda")
    output,dets= process(images,model,return_time=True)
    detection_result = []
    dets = post_process(dets,meta,num_classes,max_per_image)
    results = merge_outputs(dets,num_classes,max_per_image)
    images = images.to("cpu")
    image_detection = None
    # image_detection = np.zeros((384,384,3))
    for j in range(1, num_classes + 1):
        for bbox in results[j]:
          # print("the bbox is {}".format(bbox))
          # print("the type of bbox is     ",type(bbox))
          if bbox[4] > 0.01:
              detection_result.append([bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],j])
              image_detection = add_coco_bbox(image,bbox, j-1, conf=1, show_txt=True, img_id='default')
          # else:
          #     detection_result.append([0,0,0,0,0])
    # cv2.imshow("detection",image_detection)
    # cv2.waitKey(10)
    return image_detection,detection_result


if __name__ == '__main__':
    num_classes = 20
    max_per_image = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from models import get_pose_net
    heads = {"hm":num_classes,"wh":2,"reg":2}
    model = get_pose_net(18,heads, head_conv=64)
    # model = load_model(model,"./models/model_origin.pth")
    model = load_model(model,"./model_origin.pth")
    model.cuda()
    model.eval()

    # video = cv2.VideoCapture("t640480_det_results.avi")
    # # video = cv2.VideoCapture("MOT16-11.mp4")

    # # Exit if video not opened.
    # if not video.isOpened():
    #     print("Could not open video")
    #     sys.exit()
    
    # # Read first frame.
    # ok, frame = video.read()
    # if not ok:
    #     print('Cannot read video file')
    #     sys.exit()
    # image_result,detections = detect(frame)
    # print(detections)
    # cv2.imshow("test",image_result)
    # cv2.waitKey(0)
    # imagePath = ""
    with open("./VOC/test.txt") as f:
        lines = f.readlines()
    for line in lines:
        filename = line.rstrip()
        image = cv2.imread(filename,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_result,detections = detect_eval(image,model,num_classes,max_per_image)
        print(detections)
        if image_result is None:
            continue
        cv2.imshow("test",image_result)
        cv2.waitKey(0)

