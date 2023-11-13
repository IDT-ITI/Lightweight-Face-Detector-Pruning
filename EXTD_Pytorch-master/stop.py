from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

#imports already there
import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math

import numpy as np
from scipy.spatial import distance
#imports from face detection

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
from nni.algorithms.compression.v2.pytorch.pruning import L1NormPruner
import subprocess
#compile
os.system("python3 bbox_setup.py build_ext --inplace")
print('compile completed')

from prepare_wider_data import wider_data_file

from data.config import cfg
from EXTD_64 import build_extd
#from EXTD_32 import build_extd
from layers.modules.multibox_loss import MultiBoxLoss
from data.factory import dataset_factory, detection_collate
from logger import Logger
import wider_test
import eval_tools.evaluation as evaluation
import numpy as np
from argparse import Namespace

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Optimization options
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.') #default was 100
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.') 
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--pretrained_model', type=str, default='n40extd70.pth', help='Filename of the pretrained model')
parser.add_argument('--pruning_rate', type=float, default=0.4, help='Pruning rate for the layers')
parser.add_argument('--save_folder', type=str, default='n40stop', help='Folder to save the trained models')
parser.add_argument('--save_filename', type=str, default='n40extd', help='Base filename for saved models')


args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
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
    net = build_extd('train', cfg.NUM_CLASSES) 
    print('Load network....')
    net.load_state_dict(torch.load(f'./weights/{args.pretrained_model}'))
    net.train()
    print('Network loaded successfully')

    # define loss function (criterion) and optimizer
    criterion = MultiBoxLoss(cfg, 'face', use_cuda)


    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate,
                                 momentum=args.momentum, weight_decay=args.decay)

    
    #optimizer.load_state_dict(torch.load('./weights/optimizer.pth'))

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
    
    
    #prune the model
    pruner = L1NormPruner(net, config_list)
    pruner.compress()
    pruner._unwrap_model()

    for tepoch in range(args.epochs):
        current_learning_rate = step(optimizer, tepoch, args.gammas, args.schedule) 
        losses = 0
        train(train_loader, net, criterion, optimizer, tepoch, losses, current_learning_rate, 0)
        calc(net)
        torch.save(net.state_dict(), f'./weights/{args.save_folder}/{args.save_filename}{tepoch}.pth')



def train(train_loader, model, criterion, optimizer, epoch, losses, current_learning_rate, compress_rates_total):
    model.train()
    iteration = 0

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        if args.use_cuda:
            target = [ann.cuda() for ann in target]
            input = input.cuda()

        # compute output
        output = model(input)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_l, loss_c = criterion(output, target)
        loss = loss_l + loss_c # stress more on loss_l
        loss.backward()

        optimizer.step()

        losses += loss.item()
        end = time.time()

        if iteration % 100 == 0:
                tloss = losses / (i + 1)
                print("[epoch:{}][iter:{}][lr:{:.5f}][rate:{:.5f}] loss_class {:.8f} - loss_reg {:.8f} - total {:.8f}".format(
                    #epoch, iteration, current_learning_rate, compress_rates_total[epoch], loss_c.item(), loss_l.item(), tloss
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


def cos(optimizer, epoch, max_epochs, initial_lr, restart_epochs, final_lr):
    #Cosine annealing learning rate scheduler with restarts
    progress = epoch / max_epochs
    num_restarts = math.ceil(progress)
    epoch_in_cycle = epoch % restart_epochs
    cycle_progress = epoch_in_cycle / restart_epochs
    lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * cycle_progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


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
    subprocess.run(["python", "evaluate_all_models.py"])
