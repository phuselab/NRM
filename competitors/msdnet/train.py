from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil
import argparse

from dataloader import get_train_dataloaders_cifar10, get_train_dataloaders_stl10, get_train_dataloaders_imagenet
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model
from utils import LabelSmoothingLoss, train, validate, save_checkpoint, load_checkpoint, AverageMeter, accuracy, adjust_learning_rate

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP #JE ADDED
import torch.backends.cudnn as cudnn
import torch.optim

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--color', default='gray', type=str)
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument('--depth', default='s', type=str)
parser.add_argument('--data', default='cifar10', type=str)
parser.add_argument('--criterion', default='ce', type=str)
parser.add_argument('--num_classes', default=None, type=int)
global_args = parser.parse_args()

class MSDNETConfig():
    """
    Description:
    Configuration for MSDNET. This will be important during training. 
    
    Important changes can be selection of data='cifar10', 'cifar100', 'stl10', or 'ImageNet'
    Epochs can be changed. Default should be 300
    Refer to https://github.com/kalviny/MSDNet-PyTorch/blob/master/README.md
    """
    # global args
    tag = global_args.tag
    mode = global_args.mode
    color = global_args.color
    pretrained = global_args.pretrained
    depth = global_args.depth
    data = global_args.data
    criterion = global_args.criterion
    num_classes = global_args.num_classes
    
    #additional arguments
    arch='msdnetmod'
    batch_size=512
    criterion=criterion
    data=data
    data_root='../data/' 
    decay_rate=0.1
    epochs=100 
    evalmode=None
    evaluate_from=None
    if torch.cuda.device_count() == 4: #JE ADJUSTED
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
    save='./model_files/train/imagenet/{}_{}'.format(tag,time.strftime('%Y.%m.%d_%H.%M.%S')) # change the save path to a new path
    seed=0
    start_epoch=0
    step=2
    stepmode='even' 
    use_valid=False
    weight_decay=0.0001 
    
    workers=4 #NEED TO CHANGE DEPENDING ON COMPUTATION REQUESTS


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
        raise ValueError('invalid depth value')

def run_training():

    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
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

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model = model.cuda() #removed elif global_args.nodes > 1: model = DDP(model).cuda()
    else:
        print("NUMBER OF GPUS:",torch.cuda.device_count()) #JE ADDED
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

    if args.data.startswith('cifar'):
        train_loader, val_loader, test_loader = get_train_dataloaders_cifar10(args)
    elif args.data == 'stl10':
        train_loader, val_loader, test_loader = get_train_dataloaders_stl10(args)
    elif args.data == 'imagenet':
        train_loader, val_loader, test_loader = get_train_dataloaders_imagenet(args)
    else:
        print('NO DATASET SELECTED') 

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_prec1, train_prec5, lr = train(args, train_loader, model, criterion, optimizer, epoch)
        
        val_loss, val_prec1, val_prec5 = validate(args, val_loader, model, criterion)

        # REMOVE THIS LINE
        # train_loss, train_prec1, train_prec5, lr = val_loss, val_prec1, val_prec5, args.lr

        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            print('Best var_prec1 {}'.format(best_prec1))

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, scores)

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    ### Test the final model

    print('********** Final prediction results **********')
    validate(args, test_loader, model, criterion)

    print('finished {} epochs'.format(epoch))
    return True


if __name__ == '__main__':
    args = MSDNETConfig()
    print(vars(args))

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
            args.num_classes = 1000
        else:
            print('NO DATASET SELECTED') 

    torch.manual_seed(args.seed)

    run_training()
