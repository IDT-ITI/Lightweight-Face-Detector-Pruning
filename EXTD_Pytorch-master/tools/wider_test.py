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
import os.path as osp

import cv2
import time
import numpy as np
from PIL import Image
import scipy.io as sio

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

from tqdm import tqdm

'''
parser = argparse.ArgumentParser(description='s3fd evaluatuon wider')
parser.add_argument('--model', type=str,
                    default='weights/s3fd.pth', help='trained model')
parser.add_argument('--thresh', default=0.05, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()
'''

thresh = 0.05

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect_face(net, img, shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink,
                         interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))

    if use_cuda:
        x = x.cuda()
    # print(x.size())
    y = net(x)
    detections = y.data
    detections = detections.cpu().numpy()

    det_conf = detections[0, 1, :, 0]
    det_xmin = img.shape[1] * detections[0, 1, :, 1] / shrink
    det_ymin = img.shape[0] * detections[0, 1, :, 2] / shrink
    det_xmax = img.shape[1] * detections[0, 1, :, 3] / shrink
    det_ymax = img.shape[0] * detections[0, 1, :, 4] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= thresh)[0]
    det = det[keep_index, :]

    return det


def flip_test(net, image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(net, image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def multi_scale_test(net, image, max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(net, image, st)
    index = np.where(np.maximum(
        det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (
        st + max_im_shrink) / 2
    det_b = detect_face(net, image, bt)

    # enlarge small image x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, detect_face(net, image, bt)))
            bt *= 2
        det_b = np.row_stack((det_b, detect_face(net, image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(
            det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(
            det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets


def get_data():
    subset = 'val'
    if subset is 'val':
        wider_face = sio.loadmat(
            './eval_tools/wider_face_val.mat')
    else:
        wider_face = sio.loadmat(
            './eval_tools/wider_face_test.mat')
    event_list = wider_face['event_list']
    file_list = wider_face['file_list']
    del wider_face

    imgs_path = os.path.join(
        cfg.FACE.WIDER_DIR, 'WIDER_{}'.format(subset), 'images')
    save_path = 'eval_tools/s3fd_{}'.format(subset)

    return event_list, file_list, imgs_path, save_path


def eval_wider(model_name):
    event_list, file_list, imgs_path, save_path = get_data()
    cfg.USE_NMS = True
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(model_name))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    # transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

    counter = 0

    for index, event in enumerate(tqdm(event_list)):
        filelist = file_list[index][0]
        path = os.path.join(save_path, event[0][0].encode('utf-8'))
        if not os.path.exists(path):
            os.makedirs(path)

        for num, file in enumerate(filelist):
            im_name = file[0][0].encode('utf-8')
            in_file = os.path.join(imgs_path, event[0][0], im_name[:] + '.jpg')
            # img = cv2.imread(in_file)
            img = Image.open(in_file)
            if img.mode == 'L':
                img = img.convert('RGB')
            img = np.array(img)

            # max_im_shrink = (0x7fffffff / 577.0 /
            #                 (img.shape[0] * img.shape[1])) ** 0.5

            max_im_shrink = np.sqrt(
                1700 * 1200 / (img.shape[0] * img.shape[1]))

            shrink = max_im_shrink if max_im_shrink < 1 else 1
            counter += 1

            t1 = time.time()
            det0 = detect_face(net, img, shrink)

            det1 = flip_test(net, img, shrink)  # flip test
            [det2, det3] = multi_scale_test(net, img, max_im_shrink)

            det = np.row_stack((det0, det1, det2, det3))
            dets = bbox_vote(det)

            t2 = time.time()
            #print('Detect %04d th image costs %.4f' % (counter, t2 - t1))

            fout = open(osp.join(save_path, event[0][0].encode('utf-8'), im_name + '.txt'), 'w')
            fout.write('{:s}\n'.format(event[0][0].encode('utf-8') + '/' + im_name + '.jpg'))
            fout.write('{:d}\n'.format(dets.shape[0]))
            for i in xrange(dets.shape[0]):
                xmin = dets[i][0]
                ymin = dets[i][1]
                xmax = dets[i][2]
                ymax = dets[i][3]
                score = dets[i][4]
                fout.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'
                           .format(xmin, ymin, (xmax - xmin + 1), (ymax - ymin + 1), score))



