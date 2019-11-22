#!/usr/bin/python
# encoding: utf-8
import torch
from torch.utils.data import Dataset
from utils import read_truths_args
import random
import os
from PIL import Image
import numpy as np
import math
Debug = True
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


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy
    
    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)
    
    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        # print(labpath)
        bs = np.loadtxt(labpath)

        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))

        cc = 0
       # print("the bs is {}".format(bs))
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label

def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    
    ## data augmentation
    # img = cv2.imread(imgpath)
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    
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
    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers
       self.num_classes = 20
       self.max_objs = 50

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        if self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5 
            exposure = 1.5
            
            img, label = load_data_detection(imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
            # img.show()

        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)
    
            labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
            label = torch.zeros(50*5)
           # print(labpath)
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width).astype('float32'))
            except Exception as e:
                print(e)
                tmp = torch.zeros(1,5)
           # print(tmp)
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        mean = np.array([[[0.408,0.447,0.47 ]]])
        std = np.array([[[0.289, 0.274,0.278]]])
        img = np.array(img)
        img = ((img / 255. - mean) / std).astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32)
        # img = torch.from_numpy(img)
        # print("the shape is {}".format(img.shape))
        label = label.view(-1,5)
       # print(label)
        hm = np.zeros((self.num_classes, int(self.shape[0]/4),int(self.shape[1]/4)), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)        
        output_w = self.shape[1]//4
        output_h = self.shape[0]//4
        # print("the label is {}".format(label))
        for t in range(50):
            if label[t,1] == 0:
                break
            ct = np.array(
              [label[t,1]*output_w, label[t,2]*output_h], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            radius = gaussian_radius((math.ceil(label[t,4]*output_w), math.ceil(label[t,3]*output_w)))
            draw_umich_gaussian(hm[int(label[t,0]),:,:], ct_int, math.ceil(radius), k=1)
            wh[t] = 1. * label[t,3]*output_w, 1. * label[t,4]*output_h

            ind[t] = ct_int[1] * output_w + ct_int[0]
            reg[t] = ct - ct_int
            reg_mask[t] = 1
        ret = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, "reg":reg}
        if Debug:
            test_img = img
            # print(type(test_img))
            import cv2
            # print(test_img.shape)
            test_img = test_img.transpose(1, 2, 0)
            test_img = (test_img * std + mean)
            cv2.rectangle(test_img,(label[0,1]*384-label[0,3]*384/2,label[0,2]*384-label[0,4]*384/2),(label[0,1]*384+label[0,3]*384/2,label[0,2]*384+label[0,4]*384/2),(255, 0, 0), 2)
            cv2.imshow("tests",test_img)
            # print(label)
            # cv2.waitKey(0)
            from collections import Counter
            print(sum(hm.reshape(-1,1)==1))
            cv2.imshow("test_heatmap",hm[int(label[0,0]),:,:])
            cv2.waitKey(0)
        return (img, ret)



if __name__ == '__main__':
    # t = torch.tensor([[[1,2],[3,4],[3,4]]])
    # print(t)
    # print(t.shape)
    # # t0 = torch.gather(t, 0, torch.tensor([[[1,0],[1,1]]]))
    # # print(t0)

    # t1 = torch.gather(t, 1, torch.tensor([[[1,0],[1,1]]]))
    # print(t1)


    # t2 = torch.gather(t, 2, torch.tensor([[[1,0],[1,1]]]))
    # print(t2)

    from torchvision import datasets, transforms
    import cv2
    # opt = opts().parse()
    train_path = "E:/GazeStudy/pytorch-yolo2-master/data/VOCtrainval_06-Nov-2007/2007_train.txt"
    train_loader = torch.utils.data.DataLoader(
        listDataset(train_path, shape=(384, 384),shuffle = True, 
         train=False,seen = 0,batch_size=1),  
    batch_size=1, 
    shuffle=True,
    num_workers=1,
    pin_memory=True,  
    )
    for i,(image,label) in enumerate(train_loader):
        print(image.shape)
        # print(label.keys())
        
      
