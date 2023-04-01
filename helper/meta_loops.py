from __future__ import print_function, division

import sys
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .util import AverageMeter, accuracy, reduce_tensor, adjust_learning_rate, accuracy_list
from .optimization import find_optimal_svm


def validate(val_loader, model, criterion, opt, is_meta=False):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            if opt.dali is None:
                input, target = batch_data
            else:
                input, target = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            metrics = accuracy(output, target, topk=(1, 5))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0 and not is_meta:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
    
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret
    if is_meta: model.train()

    return top1.avg, top5.avg, losses.avg

def train_distill_multi_teacher(epoch, train_loader, module_list, criterion_list, 
                    model_s, model_s_optimizer, WeightLogits, weight_optimizer, opt):
    """One epoch distillation with multiple teacher"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    [model_t.eval() for model_t in module_list[-opt.teacher_num:]]

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    # model_t = module_list[-1]
    model_t_list = module_list[-opt.teacher_num:]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
    end = time.time()
    for idx, data in enumerate(train_loader):

        # ================prepare data==================
        data_time.update(time.time() - end)
        
        if opt.dali is None:
            input, target = data
        else:
            input, target = data[0]['data'], data[0]['label'].squeeze().long()
        input = input.float()

        if opt.gpu is not None:
            input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        feat_s, logit_s = model_s(input, is_feat=True, preact=opt.preact)
        feat_t_list = []
        logit_t_list = []
        with torch.no_grad():
            for model_t in model_t_list:
                feat_t, logit_t = model_t(input, is_feat=True, preact=opt.preact)
                feat_t = [f.detach() for f in feat_t]
                feat_t_list.append(feat_t)
                logit_t_list.append(logit_t)

        # ================compute loss====================
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        tce_loss = []
        loss_cls2 = nn.CrossEntropyLoss(reduction='none')
        for i, logit_t in enumerate(logit_t_list):
            tce_loss_temp = loss_cls2(logit_t, target)
            tce_loss.append(tce_loss_temp)
        state_logit = torch.stack(tce_loss, dim=1)

        for i, logit_t in enumerate(logit_t_list):
            state_logit = torch.cat([state_logit, logit_t], dim=1)
        loss_div_list = [criterion_div(logit_s, logit_t, is_ca=True)
                            for logit_t in logit_t_list]
        loss_div = torch.stack(loss_div_list, dim=1)
        logits_weight = WeightLogits(state_logit)
        loss_div = torch.mul(logits_weight, loss_div).sum(-1).mean()

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.zeros(1).float().cuda()

        if opt.distill_decay and epoch > (opt.epochs / 2):
            new_alpha = int(opt.epochs - epoch) /                                                         \
                int(opt.epochs / 2) * opt.alpha
            new_gamma = 1 - new_alpha

        else:
            new_alpha = opt.alpha
            new_gamma = opt.gamma
            new_beta = opt.beta

        if epoch == 1:
            loss = new_gamma * loss_cls + new_alpha * loss_div
        else:
            loss = new_gamma * loss_cls + new_alpha * loss_div + new_beta * loss_kd
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        model_s_optimizer.zero_grad()
        loss.backward()
        model_s_optimizer.step()

        
        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()


        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} \t'
                  'Acc@1 {top1.val:.3f} \t'
                  'Acc@5 {top5.val:.3f} '.format(
                      epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
        
    if opt.multiprocessing_distributed:
            # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum, data_time.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count, data_time.sum]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret
    
    return top1.avg, top5.avg, losses.avg, data_time.avg



def validate_multi(val_loader, model_list, criterion, opt):
    """validation milti model using voting"""

    model_num = len(model_list)
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_list = [AverageMeter() for i in range(model_num)]
    top1_list = [AverageMeter() for i in range(model_num)]
    top5_list = [AverageMeter() for i in range(model_num)]

    # switch to evaluate mode
    for model in model_list:
        model.eval()

    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            if opt.dali is None:
                input, target = batch_data
            else:
                input, target = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                input = input.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            output_list = []
            for model_index, model in enumerate(model_list):

                # compute output
                output = model(input)
                output_list.append(output)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses_list[model_index].update(loss.item(), input.size(0))
                top1_list[model_index].update(acc1[0], input.size(0))
                top5_list[model_index].update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % opt.print_freq == 0:
                    print(f'Model {model_index}\t'
                          'Test: [{0}/{1}]\t'
                          'GPU: {2}\t'
                          'Time: {batch_time.avg:.3f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses_list[model_index],
                              top1=top1_list[model_index], top5=top5_list[model_index]))

            acc1, acc5 = accuracy_list(output_list, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            if idx % opt.print_freq == 0:
                print('Model Ensemble\t'
                      'Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          idx, n_batch, opt.gpu, batch_time=batch_time,
                          top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        teacher_acc_top1_list = []
        teacher_acc_top5_list = []
        for model_index, model in enumerate(model_list):
            total_metrics = torch.tensor([top1_list[model_index].sum, top5_list[model_index].sum, losses_list[model_index].sum]).to(opt.gpu)
            count_metrics = torch.tensor([top1_list[model_index].count, top5_list[model_index].count, losses_list[model_index].count]).to(opt.gpu)
            total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
            count_metrics = reduce_tensor(count_metrics, 1)
            ret = []
            for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
                ret.append(s / (1.0 * n))
            teacher_acc_top1_list.append(ret[0])
            teacher_acc_top5_list.append(ret[1])
            

        ensemble_teacher_acc_top1 = reduce_tensor(torch.tensor(top1.sum)) / (1.0 * reduce_tensor(torch.tensor(top1.count)))
        ensemble_teacher_acc_top5 = reduce_tensor(torch.tensor(top5.sum)) / (1.0 * reduce_tensor(torch.tensor(top5.count)))
        
        return ensemble_teacher_acc_top1, ensemble_teacher_acc_top5, teacher_acc_top1_list


    teacher_acc_list = [t.avg for t in top1_list]
    teacher_acc_list = torch.Tensor(teacher_acc_list)


    return top1.avg, top5.avg, teacher_acc_list
