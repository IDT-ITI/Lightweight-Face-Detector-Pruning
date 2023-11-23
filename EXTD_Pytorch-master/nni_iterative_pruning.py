from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
import copy
import torch
import torch.nn.utils.prune as prune
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


#complile
os.system("python3 bbox_setup.py build_ext --inplace")
print('compile completed')

from prepare_wider_data import wider_data_file

from data.config import cfg
from EXTD_64 import build_extd
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
from logger import Logger
import wider_test
import eval_tools.evaluation as evaluation
import numpy as np
from argparse import Namespace
from nni.algorithms.compression.v2.pytorch.pruning import FPGmPruner


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

use_cuda = True

def train(net, i, start_epoch=0):

    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(cfg, 'face', use_cuda)
    train_dataset, val_dataset = dataset_factory('face')

    train_loader = data.DataLoader(train_dataset, 8,
                                num_workers=0,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=False)

    val_batchsize = 8
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                num_workers=0,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=False)

    gflops, mflops = compute_flops(net, np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE]))
    print('# of params in Classification model: %d, flops: %.2f GFLOPS, %.2f MFLOPS, image_size: %d' % \
      (sum([p.data.nelement() for p in net.parameters()]), gflops, mflops,cfg.INPUT_SIZE))

    step_index = 0
    iteration = 0
    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if use_cuda:
                images = images.cuda()
                targets = [ann.cuda()
                           for ann in targets]
            else:
                images = images
                targets = [ann for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, 0.1, step_index)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c # stress more on loss_l
            loss_add = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss_add.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                print("[epoch:{}][iter:{}][lr:{:.5f}] loss_class {:.8f} - loss_reg {:.8f} - total {:.8f}".format(
                    epoch, iteration, 1e-4, loss_c.item(), loss_l.item(), tloss
                ))

            iteration += 1

        val(epoch, net, val_loader, criterion, i)
        if iteration == cfg.MAX_STEPS:
            break
    #return net

min_loss = np.inf
def val(epoch, net, val_loader, criterion, i):
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0
    
    with torch.no_grad():
        t1 = time.time()
        for batch_idx, (images, targets) in enumerate(val_loader):
            if use_cuda:
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
            torch.save(net.state_dict(), './weights/BEST{}.pth'.format(i))
            min_loss = tloss

import numpy as np


def calc(model):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += np.count_nonzero(param.cpu().detach().numpy() == 0)

    sparsity = 100. * zero_params / total_params
    print("Model sparsity: {:.2f}%".format(sparsity))

def calc_model_sparsity(model):
    num_params = 0
    num_zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            num_params += np.prod(param.shape)
            num_zero_params += np.count_nonzero(param.cpu().detach().numpy() == 0)
    
    sparsity = num_zero_params / num_params
    
    return sparsity


config_list = [{
        'sparsity_per_layer' : 0.2,
        'op_types' : ['Conv2d'],
    }, {
        'exclude' : True,
        'op_names' : [
                    'loc.0', 'loc.1', 'loc.2', 'loc.3', 'loc.4', 'loc.5',
                    'conf.0', 'conf.1', 'conf.2', 'conf.3', 'conf.4', 'conf.5'
                    ]
    }]

def iterative_pruning_finetuning(model, num_iterations=50):
    for i in range(33, num_iterations):
        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))

        print("Pruning...")
#--------------------------------- PRUNING ------------------------------------
        pruner = FPGmPruner(model, config_list)
        pruner.compress()
        pruner._unwrap_model() 
        print('Pruning done')
        calc(model)
#--------------------------------- FINE TUNING ---------------------------------
        print("Fine-tuning...")
        train(model, i)
        torch.save(model.state_dict(), './weights/G20/G20_{}.pth'.format(i))

        pruned_model = build_extd('train', cfg.NUM_CLASSES)
        pruned_model.load_state_dict(torch.load('./weights/G20/G20{}.pth'.format(i)))
        model = pruned_model
        calc(model)
        
    return model

def main():
    # dataset setting
    print('prepare wider')
    wider_data_file()

    net = build_extd('train', cfg.NUM_CLASSES)
    print('Load network....')
    net.load_state_dict(torch.load('./weights/sfd_face.pth'), strict=True) #works also without strict=False
    print('Base network weights loaded successfully.')

    net = iterative_pruning_finetuning(model=net, num_iterations=50)

    torch.save(net.state_dict(), './weights/FINAL.pth')

if __name__ == "__main__":
    main()