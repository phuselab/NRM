#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil
import json
import argparse

from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

from dataloader import get_human_test_dataloaders_imagenet

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, smoothing=0.0):
        confidence = 1.0 - smoothing

        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def train(args, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # Import ImageNet 16 class mapping
    if (args.data=='imagenet') & (args.num_classes==16):
        sixteen_class_map_master = pd.read_csv('../datasets/imagenet_mapping.csv', header=0)
        thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values
        sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
                                        index=sixteen_class_map_master['thousand_class_id']).to_dict()

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    if args.criterion == 'ls':
        step_size = (args.num_classes-1)/(args.num_classes*(args.nBlocks-1))
        smoothings = list(reversed(list(np.arange(0.0, 1.0-1/args.num_classes + 0.01, step_size))))
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        # Filter samples to those mapped to one of 16 classes and relabel targets
        if (args.data=='imagenet') & (args.num_classes==16):
            idx_keep = torch.tensor([x.item() in thousand_class_ids for x in target])
            target = target[idx_keep]
            input = input[idx_keep]

            target = torch.tensor([sixteen_class_mapping[x.item()] for x in target])

            if input.size(0) == 0:
                continue # Skip this iteration if we filtered everything

        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if not isinstance(output, list):
            output = [output]
        
        loss = 0.0
        for j in range(len(output)):
            if args.criterion == 'ce':
                loss += criterion(output[j], target_var)
            elif args.criterion == 'ls':
                loss += criterion(output[j], target_var, smoothing=smoothings[j])

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Acc@1 {top1.val:.4f}\t'
                  'Acc@5 {top5.val:.4f}'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validateonhuman(args, model, criterion, n_flops):
    flops = n_flops
    accs = []
    results = {}
    BLOCKS_TO_USE = [0,1,2,3,4]
    for c in range(len(BLOCKS_TO_USE)):
        t = str(c)
        args.dirs = [t]
        testloader = get_human_test_dataloaders_imagenet(args)

        with torch.no_grad():
            if args.mode == 'color':
                right = dict([(k,[0,0]) for k in ['c', 'g']])
            elif args.mode == 'noise':
                right = dict([(k,[0,0]) for k in ['0', '0.04', '0.16']])
            elif args.mode == 'blur':
                right = dict([(k,[0,0]) for k in ['0', '3', '9']])

            for data in testloader:
                model.eval()
                images, labels, modes = data

                images, labels = images.cuda(), labels.cuda()
                output = model(images)

                # SHOULD FAIL AFTER THIS
                if not isinstance(output, list):
                    output = [output]

                output_t = output[BLOCKS_TO_USE[c]] # 210, 16
                print(images.size(0))
                for index in range(images.size(0)):
                    logits = output_t[index, :]
                    predict = torch.argmax(logits)
                    if predict.cpu().numpy().item() == labels[index]:
                        right[modes[index]][0] += 1

                    right[modes[index]][1] += 1
                # total += float(labels.size(0))

            # compute accuracy
            exit_accuracies = dict([(k, v[0]*100/v[1]) for k, v in right.items()])
            
            for k, v in exit_accuracies.items():
                if k not in results:
                    results[k] = []

                results[k].append(exit_accuracies[k])

            # FLOPs (only for resnet18)

            # print("caughts:", caught)
            # acceleration_ratio = 1/((0.32 * caught[0] + 0.53* caught[1] + 0.76*caught[2] + 1.0 * caught[3] + 1.07 * caught[4])/total)
            # accr_msg = "Acceleration ratio: %.4f " % acceleration_ratio

            # ops = sum([i*j for i,j in zip(net_ops, caught)])/total
            # ops_msg = 'FLOPs: %.4f ' % ops

            # print(ops_msg)
            # print(accr_msg)

            # tst_msg = 'Test Set Accuracy:  %.4f%% ' % (100 * right / total) 
            # print(tst_msg)
            # print("-----------------")
            # accs.append(100 * right / total)
            # flops.append(ops)

            # with open(os.path.join(root, "summary.log"), 'a') as f:
            #     f.write(tst_msg + accr_msg + ops_msg + '\n')

    # print('flops = {}'.format(flops))
    # print('accs = {}'.format(accs))
    print(results)
    results['flops'] = flops
    
    with open(os.path.join(args.save, 'results.json'), 'w') as fp:
        print(args.save)
        json.dump(results, fp)


def validate_force_flops(args, val_loader, model, criterion):
    #ADD IN DYANMIC INFERENCE -- we have args
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # Import ImageNet 16 class mapping
    if (args.data=='imagenet') & (args.num_classes==16):
        sixteen_class_map_master = pd.read_csv('../datasets/imagenet_mapping.csv', header=0)
        thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values
        sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
                                        index=sixteen_class_map_master['thousand_class_id']).to_dict()

    model.eval()
    flops = torch.load(os.path.join(args.save, 'flops.pth'))

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # Filter samples to those mapped to one of 16 classes and relabel targets
            if (args.data=='imagenet') & (args.num_classes==16):
                idx_keep = torch.tensor([x.item() in thousand_class_ids for x in target])
                target = target[idx_keep]
                input = input[idx_keep]

                target = torch.tensor([sixteen_class_mapping[x.item()] for x in target])

                if input.size(0) == 0:
                    continue # Skip this iteration if we filtered everything

            target = target.cuda() 
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            accs = [[] for _ in range(len(output))]
            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                accs[j].append(prec1.item())
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    avg_accs = [np.mean(a) for a in accs]
    print('FLOPS: {}\nACCS: {}'.format(flops, avg_accs))
    return losses.avg, top1[-1].avg, top5[-1].avg


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # Import ImageNet 16 class mapping
    if (args.data=='imagenet') & (args.num_classes==16):
        sixteen_class_map_master = pd.read_csv('../datasets/imagenet_mapping.csv', header=0)
        thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values
        sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
                                        index=sixteen_class_map_master['thousand_class_id']).to_dict()

    model.eval()

    tracker_ks = 0

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            # Filter samples to those mapped to one of 16 classes and relabel targets
            if (args.data=='imagenet') & (args.num_classes==16):
                idx_keep = torch.tensor([x.item() in thousand_class_ids for x in target])
                target = target[idx_keep]
                input = input[idx_keep]

                target = torch.tensor([sixteen_class_mapping[x.item()] for x in target])

                if input.size(0) == 0:
                    continue # Skip this iteration if we filtered everything

            target = target.cuda()
            input = input.cuda()         

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg

def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
