#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import argparse
import shutil

from dataloader import get_human_test_dataloaders_imagenet, get_test_dataloaders_cifar10, get_test_dataloaders_stl10, get_test_dataloaders_imagenet
from adaptive_inference import dynamic_evaluate, anytime_evaluate
import models
from op_counter import measure_model
from utils import LabelSmoothingLoss, train, validate, validate_force_flops, save_checkpoint, load_checkpoint, AverageMeter, accuracy, adjust_learning_rate, validateonhuman

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

parser = argparse.ArgumentParser(description='PyTorch Imagenet Training') 
parser.add_argument('--load_path', type=str)
parser.add_argument('--color', default='gray', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--tag', type=str)
parser.add_argument('--sweep-step', default=1, type=int)
parser.add_argument('--depth', default='s', type=str)
parser.add_argument('--data', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--criterion', default='ce', type=str)
parser.add_argument('--evalmode', default='force_flops', type=str)
parser.add_argument('--anytime_threshold', default=0.9, type=float)
parser.add_argument('--inference', default='standard', type=str)
parser.add_argument('--subsample', default=False, type=bool)
global_args = parser.parse_args()

class MSDNETConfig():
    """
    Description:
    Configuration for MSDNET. This will be important during training. 
    
    Important changes can be selection of data='cifar10' or 'cifar100'
    Epochs can be changed. Default should be 300
    Refer to https://github.com/kalviny/MSDNet-PyTorch/blob/master/README.md
    """
    # global args
    tag = global_args.tag
    mode = global_args.mode
    color = global_args.color
    load_path = global_args.load_path  
    sweep_step = global_args.sweep_step
    depth = global_args.depth
    data = global_args.data
    criterion = global_args.criterion
    evalmode = global_args.evalmode
    anytime_threshold = global_args.anytime_threshold
    num_classes = global_args.num_classes
    inference = global_args.inference
    subsample = global_args.subsample

#additional args
    arch='msdnetmod'
    batch_size=1024
    criterion=criterion
    data=data
    data_root='../data/'
    decay_rate=0.1
    epochs=100
    evalmode=evalmode
    evaluate_from=load_path
    if torch.cuda.device_count() == 4:
        gpu='0,1,2,3'
    elif torch.cuda.device_count() == 2:
        gpu='0, 1' 
    else:
        gpu='0'
    lr=0.1
    lr_type='multistep'
    momentum=0.9
    optimizer='sgd'
    print_freq=10
    prune='max' 
    reduction=0.5
    resume=False
    save='./model_files/testonhuman/imagenet/{}_{}'.format(tag,time.strftime('%Y.%m.%d_%H.%M.%S')) # change the save path to a new path
    seed=0
    start_epoch=0
    step=2
    stepmode='even' 
    use_valid=False
    weight_decay=0.0001 
    workers=4
    
    
    # medium and large model (for backbone comparison)
    if depth == 'm' or depth == 'l':
        base=4
        bnFactor='1-2-4' 
        bottleneck=True 
        grFactor='1-2-4' 
        growthRate=6
        nBlocks=7
        nChannels=16
    # smallest model (from paper)
    elif depth == 's':
        base=3
        bnFactor='1-1-1' # changed
        bottleneck=False          
        grFactor='1-1-1' # changed
        growthRate=6
        nBlocks=7 # changed
        nChannels=8 # changed
    else:
        raise ValueError("invalid depth value")


def run_testing():

    global args
    epoch, best_prec1, best_epoch = 0, 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data == 'cifar10' or args.data == 'cifar100':
        IM_SIZE = 32
    elif args.data == 'stl10':
        IM_SIZE = 96
    elif args.data == 'imagenet':
        IM_SIZE = 224
    else:
        print('NO DATASET SELECTED') 

    model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)    
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del(model)
        
        
    model = getattr(models, args.arch)(args)
    model = torch.nn.DataParallel(model).cuda()

    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.criterion == 'ls':
        criterion = LabelSmoothingLoss(args.num_classes).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        validateonhuman(args, model, criterion, n_flops)
        return
    else:
        print('NEED EVALUTAION MODE')

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    ### Test the final model

    print('********** Final prediction results **********')
    validate(test_loader, model, criterion)

    print('finished {} epochs'.format(epoch))
    return True


if __name__ == "__main__":
    # args = arg_parser.parse_args()

    args = MSDNETConfig()

    if args.mode == 'noise':
        noise_stds = list(np.arange(0.09, 0.28, 0.02))
        args.noise = noise_stds[args.sweep_step - 1]
        args.save = args.save + '_noise' + str(args.noise)
        print("NOISE:", str(args.noise))
    elif args.mode == 'blur':
        blur_stds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
        args.blur = blur_stds[args.sweep_step - 1]
        args.save = args.save + '_blur' + str(args.blur)
    elif args.mode == 'color':
        args.save = args.save + '_color_' + str(args.color)

    print(args) 
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.nScales = len(args.grFactor)

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.num_classes is None:
        if args.data == 'cifar10' or args.data == 'stl10':
            args.num_classes = 10
        elif args.data == 'cifar100':
            args.num_classes = 100
        elif args.data == 'imagenet':
            args.num_classes = 16
        else:
            print('NO DATASET SELECTED') 

    torch.manual_seed(args.seed)

    run_testing()
