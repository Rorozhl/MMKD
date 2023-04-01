from __future__ import print_function
from numpy import append
from numpy.core.fromnumeric import transpose

import numpy as np
import torch, copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from distiller_zoo import DistillKL

class LogitsWeight(nn.Module):
    def __init__(self, n_feature, teacher_num, factor=2):
        super(LogitsWeight, self).__init__()
        self.layer = nn.Linear(n_feature, n_feature//factor, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.all_act = nn.Linear(n_feature//factor, teacher_num)

    def forward(self, logit_t_list, logit_s):
        x = torch.cat(logit_t_list, dim=1)
        x = torch.cat((x, logit_s), dim=1)
        # for i, logit_t in enumerate(logit_t_list):
        #     x = torch.cat([x, logit_t], dim=1)
        x = self.relu(self.layer(x))
        x = self.all_act(x)
        return torch.softmax(x, dim=-1) 

class FeatureWeight(nn.Module):
    def __init__(self, bs, teacher_num, factor=2):
        super(FeatureWeight, self).__init__()
        n_feature = bs * (teacher_num + 1)
        self.layer = nn.Linear(n_feature, n_feature//factor, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.all_act = nn.Linear(n_feature//factor, teacher_num)
        self.batch_size = bs

    def forward(self, feat_t_list, feat_s):
        state_feat = []
        for i in range(len(feat_t_list)):
            sim_temp = feat_t_list[i].reshape(self.batch_size, -1)
            sim_t = torch.matmul(sim_temp, sim_temp.t())
            state_feat.append(sim_t)
        x = torch.cat(state_feat, dim=1)
        sim_temp = feat_s.reshape(self.batch_size, -1)
        sim_t = torch.matmul(sim_temp, sim_temp.t())
        x = torch.cat((x, sim_t), dim=1)
        # for i, logit_t in enumerate(logit_t_list):
        #     x = torch.cat([x, logit_t], dim=1)
        x = self.relu(self.layer(x))
        x = self.all_act(x)
        return torch.softmax(x, dim=-1)               

class MatchLogits(nn.Module):
    def __init__(self, opt):
        super(MatchLogits, self).__init__()
        self.criterion_div = DistillKL(opt.kd_T) 

    def forward(self, logit_s, logit_t_list, logits_weight):
        loss_div_list = [self.criterion_div(logit_s, logit_t, is_ca=True)
                            for logit_t in logit_t_list]
        loss_div = torch.stack(loss_div_list, dim=1)
        loss_div = torch.mul(logits_weight, loss_div).sum(-1).mean()
        return loss_div

class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, factor=2, convs=False):
        super(Embed, self).__init__()
        self.convs = convs
        if self.convs:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_in//factor, kernel_size=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in//factor, dim_in//factor, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True), 
                nn.Conv2d(dim_in//factor, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)              
            )
        else:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True) 
            )


    def forward(self, x):
        x = self.transfer(x)
        return x


class MatchFeature(nn.Module):
    def __init__(self, t_len, s_n, t_n, convs=False): 
        super(MatchFeature, self).__init__()
          
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.crit = nn.MSELoss(reduction='none').cuda()
        
        for i in range(t_len):
            setattr(self, 'regressor'+str(i), Embed(s_n, t_n[i], convs))
               
    def forward(self, feat_s, feat_t_list, weight):
        bsz = feat_s.shape[0]
        # feature space alignment
        proj_s = []
        proj_t = []
        s_H = feat_s.shape[2]
        for i in range(len(feat_t_list)):
            t_H = feat_t_list[i].shape[2]
            if s_H > t_H:
                input = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
                proj_s.append(getattr(self, 'regressor'+str(i))(input))
                proj_t.append(feat_t_list[i])
            elif s_H < t_H or s_H == t_H:
                target = F.adaptive_avg_pool2d(feat_t_list[i], (s_H, s_H))
                proj_s.append(getattr(self, 'regressor'+str(i))(feat_s))
                proj_t.append(target)
        
        ind_loss = torch.zeros(bsz, len(feat_t_list)).cuda()

        for i in range(len(feat_t_list)):
            ind_loss[:, i] = self.crit(proj_s[i], proj_t[i]).reshape(bsz, -1).mean(-1)

        loss = (weight.cuda() * ind_loss).sum()/(1.0*bsz)
        return loss
        
