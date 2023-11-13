'''
EXTD Copyright (c) 2019-present NAVER Corp. MIT License
'''

#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


#complie
os.system("python3 bbox_setup.py build_ext --inplace")
print('compile completed')

from prepare_wider_data import wider_data_file

from data.config import cfg
from EXTD_32 import build_extd
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
from logger import Logger
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


try:
    import nsml
    USE_NSML=True
except ImportError:
    USE_NSML=False
    from wider_test import eval_wider

USE_NSML=False
from wider_test import eval_wider

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

print('argparse')
parser = argparse.ArgumentParser(
    description='EXTD face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset',
                    default='face',
                    choices=['hand', 'face', 'head'],
                    help='Train target')
parser.add_argument('--basenet',
                    # default='vgg16_reducedfc.pth',
                    default='/m2/gkrispanis/Projects/EXTD_Pytorch-master2/weights/nni_res.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=str2bool,
                    help='Use mutil Gpu training')
parser.add_argument('--eval_verbose',
                    default=True, type=str2bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--pretrained', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

args = parser.parse_args()

def compute_flops(model, image_size):
  import torch.nn as nn
  flops = 0.
  input_size = image_size
  for m in model.modules():
    if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
      input_size = input_size / 2.
    if isinstance(m, nn.Conv2d):
      if m.groups == 1:
        flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]) * m.kernel_size[0] ** 2 * m.in_channels * m.out_channels
      else:
        flop = (input_size[0] / m.stride[0] * input_size[1] / m.stride[1]) * m.kernel_size[0] ** 2 * ((m.in_channels/m.groups) * (m.out_channels/m.groups) * m.groups)
      flops += flop
      if m.stride[0] == 2: input_size = input_size / 2.

  return flops / 1000000000., flops / 1000000




# dataset setting
print('prepare wider')
wider_data_file()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


train_dataset, val_dataset = dataset_factory(args.dataset)

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=False, generator=torch.Generator(device='cuda'))

val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=False)

min_loss = np.inf
start_epoch = 0
s3fd_net = build_extd('train', cfg.NUM_CLASSES)
net = s3fd_net

print(net)
gflops, mflops = compute_flops(net, np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE]))
print('# of params in Classification model: %d, flops: %.2f GFLOPS, %.2f MFLOPS, image_size: %d' % \
      (sum([p.data.nelement() for p in net.parameters()]), gflops, mflops,cfg.INPUT_SIZE))



if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    start_epoch = net.load_weights(args.resume)

else:
    try:
        #_weights = torch.load(args.pretrained)
        #print('Load base network....')
        print('Load network....')
        net.load_state_dict(torch.load(args.pretrained), strict=True) #works also without strict=False
        #net.base.load_state_dict(_weights['state_dict'], strict=False)
        print('Base network weights loaded successfully.')
    except:
        print('Failed to load base network weights. Initializing base network....')
        #print('Initialize base network....')
        net.base.apply(net.weights_init)

if args.cuda:
    if args.multigpu:
        net = torch.nn.DataParallel(net)
    net = net.cuda()
    cudnn.benckmark = True

if not args.resume:
    print('Initializing weights...')
    #s3fd_net.extras.apply(s3fd_net.weights_init) # used only for s3fd
    s3fd_net.loc.apply(s3fd_net.weights_init)
    s3fd_net.conf.apply(s3fd_net.weights_init)
    #s3fd_net.head.apply(s3fd_net.weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)

print('Loading wider dataset...')
print('Using the specified args:')
print(args)

tensor_board_dir = os.path.join('./logs', 'tensorboard')
if not os.path.isdir(tensor_board_dir):
    os.mkdir(tensor_board_dir)
logger = Logger(tensor_board_dir)

#prune the model


#pruner = FPGMPruner(net, config_list)
#pruner.compress()


def train():
    step_index = 0
    iteration = 0
    net.train()

    #accumulation_steps = 2  # Accumulate gradients over two iterations

    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            out = net(images)

            # backprop
            #optimizer.zero_grad() if (iteration % accumulation_steps) == 0 else None
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss_add = loss_l + loss_c
            loss.backward()
            optimizer.step()

            #if (iteration + 1) % accumulation_steps == 0:
            #    optimizer.step()

            t1 = time.time()
            losses += loss_add.item()

            if USE_NSML:
                if iteration % 10 == 0:
                    nsml.report(
                        step=iteration,
                        train_loss_classification=loss_c.item(),
                        train_loss_regression=loss_l.item(),
                        train_total_loss=loss_add.item(),
                    )

            else:
                if iteration % 10 == 0:
                    logger.scalar_summary('train_loss_classification', loss_c.item(), iteration, scope='c')
                    logger.scalar_summary('train_loss_regression', loss_l.item(), iteration, scope='r')
                    logger.scalar_summary('train_total_loss', loss_add.item(), iteration, scope='t')

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                print("[epoch:{}][iter:{}][lr:{:.5f}] loss_class {:.8f} - loss_reg {:.8f} - total {:.8f}".format(
                    epoch, iteration, args.lr, loss_c.item(), loss_l.item(), tloss
                ))

            iteration += 1

        val(epoch)
        if iteration == cfg.MAX_STEPS:
            break

    #pruner._unwrap_model()
    #torch.save(net.state_dict(), './weights/SCRATCH.pth')


def val(epoch):
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0

    with torch.no_grad():
        t1 = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.cuda:
                images = images.cuda()

                targets = [ann.cuda() for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            out = net(images)
            loss_l, loss_c = criterion(out, targets)

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            step += 1

        tloss = (loc_loss + conf_loss) / step
        t2 = time.time()
        print('Timer: %.4f' % (t2 - t1))
        print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

        global min_loss
        if tloss < min_loss:
            print('Saving best state,epoch', epoch)
            #file = 'sfd_{}.pth'.format(args.dataset)
            torch.save(net.state_dict(), './weights/32.pth')
            torch.save(optimizer.state_dict(), './weights/optimizer64.pth')
            #torch.save(s3fd_net.state_dict(), os.path.join(
            #    args.save_folder, file))
            min_loss = tloss

        states = {
            'epoch': epoch,
            'weight': net.state_dict(),
        }
        #file = 'sfd_{}_checkpoint.pth'.format(args.dataset)
        torch.save(states, './weights/checkpoint.pth')
        #torch.save(states, os.path.join(
        #    args.save_folder, file))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
    #if USE_NSML==False:
    #    eval_wider('/m2/gkrispanis/Projects/EXTD_Pytorch-master2/weights/test_pruning_re2.pth')

