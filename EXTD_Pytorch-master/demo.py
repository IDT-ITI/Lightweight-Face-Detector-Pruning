'''
EXTD Copyright (c) 2019-present NAVER Corp. MIT License
'''
#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from EXTD_64 import build_extd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

import tqdm


parser = argparse.ArgumentParser(description='EXTD demo')
parser.add_argument('--save_dir', type=str, default='tmp/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/EXTD_64.pth', help='trained model')
parser.add_argument('--thresh', default=0.6, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh, wid, hei):
    #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape


    if wid > 0 and hei > 0:
        image = cv2.resize(img, (wid, hei))
    else:
        max_im_shrink = np.sqrt(1700 * 1200 / (img.shape[0] * img.shape[1]))
        image = cv2.resize(img, None, None, fx=max_im_shrink, fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)


    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()


    with torch.no_grad():

        t1 = time.time()

        y = net(x)
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
                j += 1
                cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
                conf = "{:.3f}".format(score)
                point = (int(left_up[0]), int(left_up[1] - 5))
                # cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,
                #            0.6, (0, 255, 0), 1)

        t2 = time.time()



    #print('detect:{} timer:{}'.format(img_path, t2 - t1))

    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)

    return t2-t1


if __name__ == '__main__':
    net = build_extd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_path = './img'
    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]

    for path in img_list:
        dt = detect(net, path, args.thresh, -1, -1)

    '''
    file = open('time_check.txt', 'w')


    for ind in range(51):
        t = 0
        for rnd in tqdm.tqdm(range(1500)):
            img_path = './img'
            img_list = [os.path.join(img_path, x)
                        for x in os.listdir(img_path) if x.endswith('jpg')]

            for path in img_list:
                dt = detect(net, path, args.thresh, 280 + 40 * ind, 280 + 40 * ind)
                if rnd>500:
                    t += dt



        print('{} {}'.format(280 + 40 * ind, t/1000))
        file.write('{} {}\n'.format(280 + 40 * ind, t/1000))

    file.close()
    '''
