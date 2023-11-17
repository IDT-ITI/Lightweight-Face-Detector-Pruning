from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

#imports already there
import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import math

import numpy as np
import os
import copy
import torch
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
import torch.backends.cudnn as cudnn
from nni.algorithms.compression.v2.pytorch.pruning import FPGMPruner, L1NormPruner
import subprocess

#compile
os.system("python3 ./eval_tools/bbox_setup.py build_ext --inplace")
print('compile completed')

from prepare_wider_data import wider_data_file

from data.config import cfg
from models.eresfd import build_model
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
import wider_test
import eval_tools.evaluation as evaluation
import numpy as np
from argparse import Namespace

parser = argparse.ArgumentParser(description='Stop pruning and fine tune the model without changing pruned weights',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Optimization options
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[5],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--pretrained_model', type=str, default='./weights/40eRes200.pth', help='path to the pretrained model')
parser.add_argument('--pruned_eres', type=str, default='./weights/g40stop', help='path to save pruned weights without epoch number')
parser.add_argument('--pruning_rate', type=float, default=0.2, help='sparsity per layer')

parser.add_argument('--epoch_prune', type=int, default=5, help='compress layer of model')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
args.manualSeed = 42

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True
use_cuda = True


def main():

    # Init dataset
    print('prepare wider')
    wider_data_file()

    train_dataset, _ = dataset_factory('face')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, collate_fn=detection_collate, pin_memory=False, generator=torch.Generator(device='cuda')) 

    # Init model, criterion, and optimizer
    net = build_model('train', cfg.NUM_CLASSES, width_mult=0.0625) 
    print('Load network....')
    net.load_state_dict(torch.load(args.pretrained_model))
    print('Network loaded successfully')

    # define loss function (criterion) and optimizer
    criterion = MultiBoxLoss(cfg, 'face', use_cuda)


    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.decay)


    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    config_list = [{
        'sparsity_per_layer' : args.pruning_rate,
        'op_types' : ['Conv2d'],
    }, {
        'exclude' : True,
        'op_names' : [
                    'loc.0', 'loc.1', 'loc.2', 'loc.3', 'loc.4', 'loc.5',
                    'conf.0', 'conf.1', 'conf.2', 'conf.3', 'conf.4', 'conf.5'
                    ]
    }]

    pruner = FPGMPruner(net, config_list)
    pruner.compress()
    pruner._unwrap_model()
    print('Model pruned')

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = step(optimizer, epoch, args.gammas, args.schedule)
        train(train_loader, net, criterion, optimizer, epoch, 0, current_learning_rate, args.pruning_rate)
        calc(net)
        torch.save(net.state_dict(), './weights/n40stop/n40stopRes{}.pth'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, losses, current_learning_rate, compress_rates_total):
    model.train()
    iteration = 0

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        if args.use_cuda:
            targets = [ann.cuda() for ann in target]
            input = input.cuda()

        # compute output
        output = model(input)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_l, loss_c = wrapper(input, model, criterion, targets)
        loss = loss_l + loss_c # stress more on loss_l
        loss.backward()

        optimizer.step()

        losses += loss.item()
        end = time.time()

        if iteration % 100 == 0:
                tloss = losses / (i + 1)
                print("[epoch:{}][iter:{}][lr:{:.5f}][rate:{:.5f}] loss_class {:.8f} - loss_reg {:.8f} - total {:.8f}".format(
                    epoch, iteration, current_learning_rate, compress_rates_total, loss_c.item(), loss_l.item(), tloss
                ))

        iteration += 1


def step(optimizer, epoch, gammas, schedule):
    #Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def wrapper(images, net, criterion, targets):
    outputs = net(images)
    total_loss_l = 0
    total_loss_c = 0
    for out in outputs:
        loss_l, loss_c = criterion(out, targets)
        total_loss_l += loss_l
        total_loss_c += loss_c
    return total_loss_l, total_loss_c

def calc(model):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += np.count_nonzero(param.cpu().detach().numpy() == 0)

    sparsity = 100. * zero_params / total_params
    print("Model sparsity after training: {:.2f}%".format(sparsity))


if __name__ == '__main__':
    main()
