from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import os
import math
import numpy as np
import pandas as pd


def anytime_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save, 'logits_single.pth')): 
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save, 'logits_single.pth')) 
    else: 
        if (args.data=='ImageNet') & (args.num_classes==1000) & (args.inference=='sixteen'):
            val_pred, val_target = tester.calc_logit_thousand_to_sixteen(val_loader) 
            test_pred, test_target = tester.calc_logit_thousand_to_sixteen(test_loader) 
        else:
            val_pred, val_target = tester.calc_logit(val_loader) 
            test_pred, test_target = tester.calc_logit(test_loader)
        torch.save((val_pred, val_target, test_pred, test_target), 
                    os.path.join(args.save, 'logits_single.pth'))

    flops = torch.load(os.path.join(args.save, 'flops.pth'))

    with open(os.path.join(args.save, 'anytime.txt'), 'w') as fout:
        if args.anytime_threshold != None:
            T = [args.anytime_threshold] * (test_pred.size()[0]-1) + [-1e8]
            print("*********************")
            acc_test, exp_flops = tester.dynamic_eval_with_threshold(
                test_pred, test_target, flops, T, args)
            print('test acc: {:.3f}, test flops: {:.2f}'.format(acc_test, exp_flops / 1e6))
            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))
        else:
            for T in range(1, 10):
                T = [0.1 * T] * (test_pred.size()[0]-1) + [-1e8]
                print("*********************")
                acc_test, exp_flops = tester.dynamic_eval_with_threshold(
                    test_pred, test_target, flops, T, args)
                print('test acc: {:.3f}, test flops: {:.2f}'.format(acc_test, exp_flops / 1e6))
                fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))

def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save, 'logits_single.pth')): 
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save, 'logits_single.pth')) 
    else: 
        if (args.data=='ImageNet') & (args.num_classes==1000) & (args.inference=='sixteen'): #special case for 16 class inference when imagenet model trains on 1000
            val_pred, val_target = tester.calc_logit_thousand_to_sixteen(val_loader) 
            test_pred, test_target = tester.calc_logit_thousand_to_sixteen(test_loader) 
        else:
            val_pred, val_target = tester.calc_logit(val_loader) 
            test_pred, test_target = tester.calc_logit(test_loader) 
        torch.save((val_pred, val_target, test_pred, test_target), 
                    os.path.join(args.save, 'logits_single.pth'))

    flops = torch.load(os.path.join(args.save, 'flops.pth'))

    with open(os.path.join(args.save, 'dynamic.txt'), 'w') as fout:
        for p in range(1, 40):
            print("*********************")
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            probs = torch.exp(torch.log(_p) * torch.range(1, args.nBlocks))
            probs /= probs.sum()
            acc_val, _, T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs, flops, args)
            acc_test, exp_flops = tester.dynamic_eval_with_threshold(test_pred, test_target, flops, T, args)
            print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}M'.format(acc_val, acc_test, exp_flops / 1e6))
            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))


class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = self.args.nBlocks
        logits = [[] for _ in range(n_stage)]
        targets = []

        # Import mapping for ImageNet 16 classes
        if (self.args.data=='ImageNet') & (self.args.num_classes==16):
            sixteen_class_map_master = pd.read_csv('imagenet_mapping.csv', header=0)
            thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values
            sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values,
                                            index=sixteen_class_map_master['thousand_class_id']).to_dict()

        for i, (input, target) in enumerate(dataloader):
            # Filter samples to those that map to one of the 16 classes and relabel targets
            if (self.args.data=='ImageNet') & (self.args.num_classes==16):
                idx_keep = torch.tensor([x.item() in thousand_class_ids for x in target])
                target = target[idx_keep]
                input = input[idx_keep]

                target = torch.tensor([sixteen_class_mapping[x.item()] for x in target])

            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t) 

            if i % self.args.print_freq == 0: 
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    #adding method to do inference on 16 classes by summing logits of super-classes
    def calc_logit_thousand_to_sixteen(self, dataloader):

        #add in mapping for 16 class inference
        sixteen_class_map_master = pd.read_csv('imagenet_mapping.csv', header=0) #read in mapping csv
        thousand_class_ids = sixteen_class_map_master['thousand_class_id'].values #array of values to keep
        sixteen_class_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values, 
            index=sixteen_class_map_master['thousand_class_id']).to_dict() #dictionary mapping to new classes
        reduced_mapping = pd.Series(sixteen_class_map_master['sixteen_class_id'].values, 
            index=sixteen_class_map_master['mid_map']).to_dict() #dictionary for maps based on reduced output space
        
        self.model.eval()
        n_stage = self.args.nBlocks
        logits = [[] for _ in range(n_stage)]
        targets = []
        for i, (input, target) in enumerate(dataloader):
            idx_keep = torch.tensor([x.item() in thousand_class_ids for x in target]) #list of indices to keep
            target = target[idx_keep] #narrow down target list
            input = input[idx_keep] #narrow down input list
            target = torch.tensor([sixteen_class_mapping[x.item()] for x in target]) #map new target
            
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t) 

            if i % self.args.print_freq == 0: 
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))
        
        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        
        sixteen_class_map_master = sixteen_class_map_master.sort_values(by='sixteen_class_id') #sort values of 16 class indeces
        idx_keep = sixteen_class_map_master['thousand_class_id'].values #get the indices corresponding to 1000 classes
        ts_logits = ts_logits[:,:,idx_keep] #filter on these indices
        
        #split up array by 16 class index sum up the logits and rejoin
        ts_logits_splits = ts_logits.split([1,4,2,49,5,7,3,6,4,3,109,2,2,1,1,8], dim=2)
        cross_logits = []
        for split in ts_logits_splits:
            cross_logits.append(split.sum(dim=2)) 

        ts_logits = torch.stack(cross_logits, dim=2)

        return ts_logits, ts_targets


    def dynamic_eval_find_threshold(self, logits, targets, p, flops, args):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item()) 

                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T, args):
        
        n_stage, n_sample, _ = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item()) 
                    
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops 