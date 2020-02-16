#!/usr/bin/python
# encoding: utf-8
import torch
from torch.utils.data import Dataset
import random
import os
from PIL import Image
import numpy as np
import math
import xml.etree.ElementTree as ET
from image import color_aug,flip,get_affine_transform,affine_transform
import cv2

Debug = False 

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)    
    x, y = int(center[0]), int(center[1])  
    height, width = heatmap.shape[0:2]
      
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)  

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def load_label(label_path):
    tree=ET.parse(label_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    label = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = [cls_id,float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
        label.append(b)
    return label


def load_data_detection(imgpath, shape):
    label_path = imgpath.replace('images', 'labels').replace('JPEGImages', 'Annotations').replace('.jpg', '.xml').replace('.png','.xml')
    img = cv2.imread(imgpath)
    label = load_label(label_path)
    return img,label


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, train=False):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.train = train
       self.shape = shape
       self.num_classes = 20
       self.max_objs = 100
       self.not_rand_crop = False
       self.flip = True
       self.no_color_aug = False
       self._data_rng = np.random.RandomState(123)
       self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
       self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
          ], dtype=np.float32)

       self.mean = np.array([0.485, 0.456, 0.406],
            dtype=np.float32).reshape(1, 1, 3)
       self.std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return self.nSamples

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        img, label = load_data_detection(imgpath, self.shape)
        testlabel = label.copy()
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.shape[0], self.shape[1]
    
        flipped = False
        if self.train:
            if not self.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

      
            if np.random.random() < self.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1
        
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, 
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
            
        inp = (inp.astype(np.float32) / 255.)
        if self.train and not self.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        output_h = input_h // 4
        output_w = input_w // 4
 
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        img = np.array(inp)
        img = ((img - self.mean) / self.std).astype(np.float32)

        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32)

        hm = np.zeros((self.num_classes, int(self.shape[0]/4),int(self.shape[1]/4)), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)        


        label = np.array(label)
        for k in range(label.shape[0]):
          
            cls_id = int(label[k,0])
            bbox = label[k,1:5]
          
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                 [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

        ret = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, "reg":reg}       
        if Debug:
            testlabel = np.array(testlabel)
            cv2.rectangle(test_img,(int(testlabel[0,1]),int(testlabel[0,2])),(int(testlabel[0,3]),int(testlabel[0,4])),(255, 0, 0), 2)
            cv2.imshow("heatmap",hm[int(label[0,0])])
            cv2.imshow("tests",test_img)

            print(label)
            cv2.waitKey(0)
        return (img, ret)



if __name__ == '__main__':
    from torchvision import datasets, transforms
    import cv2
    train_path = "./VOC/train.txt"
    train_loader = torch.utils.data.DataLoader(
        listDataset(train_path, shape=(384, 384),shuffle = True, 
         train=True),  
    batch_size=1, 
    shuffle=True,
    num_workers=0,
    pin_memory=True,  
    )
    for i,(image,label) in enumerate(train_loader):
        print(image.shape)
        # print(label.keys())
        
      
