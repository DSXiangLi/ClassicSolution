# -*-coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def ramp_up(cur_epoch, max_epoch, method):
    """
    根据训练epoch来调整无标注loss部分的权重，初始epoch无标注loss权重为0
    """

    def linear(cur_epoch, max_epoch):
        return cur_epoch / max_epoch

    def sigmoid(cur_epoch, max_epoch):
        p = 1.0 - cur_epoch / max_epoch
        return np.exp(-5.0 * p ** 2)

    def cosine(cur_epoch, max_epoch):
        p = cur_epoch / max_epoch
        return 0.5 * (np.cos(np.pi * p) + 1)

    if cur_epoch == 0:
        weight = 0.0
    else:
        if method == 'linear':
            weight = linear(cur_epoch, max_epoch)
        elif method == 'sigmoid':
            weight = sigmoid(cur_epoch, max_epoch)
        elif method == 'cosine':
            weight = cosine(cur_epoch, max_epoch)
        else:
            raise ValueError('Only linear, sigmoid, cosine method are supported')
    return weight


class TemporalEnsemble(object):
    def __init__(self, tp):
        self.loss_fn = tp.loss_fn
        self.wmax = tp.max_unsupervised * tp.labeled_size / tp.num_train_steps  # resacle the rmse loss
        self.alpha = torch.tensor(tp.temporal_alpha)
        self.Z = torch.zeros(tp.num_train_steps, tp.label_size).float()
        self.target = torch.zeros(tp.num_train_steps, tp.label_size).float()
        self.output = torch.zeros(tp.num_train_steps, tp.label_size).float()
        self.epoch_size = tp.epoch_size
        self.cur_epoch = 0
        self.ramp_up_method = tp.ramp_up_method

    def epoch_update(self):
        # update after each epoch
        self.Z = self.alpha * self.Z + (1 - self.alpha) * self.output
        self.target = self.Z / (1 - torch.pow(self.alpha, self.cur_epoch + 1))  # startup bias
        self.cur_epoch += 1

    def compute_loss(self, features, logits):
        self.output[features['idx']] = logits.detach()  # 注意不要直接赋值，需要detach这部分不做梯度更新
        ## masked supervised loss
        labels = features['label']
        cond = labels >= 0
        self.supervised_loss = self.loss_fn(logits[cond], labels[cond])

        ## temporal loss
        z_i = self.target[features['idx']]
        self.mse_loss = torch.mean((F.softmax(logits, dim=1) - F.softmax(z_i, dim=1)) ** 2)

        ## weight
        weight = ramp_up(self.cur_epoch, self.epoch_size, self.ramp_up_method) * self.wmax
        loss = self.supervised_loss + weight * self.mse_loss
        return loss


def create_ema_model(model):
    for param in model.parameters():
        param.detach_()
    return model


class MeanTeacher(object):
    def __init__(self, model, ema_model, tp, tb):
        self.model = model
        self.ema_model = ema_model
        self.tp = tp
        self.tb = tb
        self.step = 0
        self.epoch = 0
        self.log_step = self.tp.log_step
        self.num_train_steps = tp.num_train_steps
        self.loss_fn = tp.loss_fn
        self.epoch_size = tp.epoch_size
        self.ramp_up_method = tp.ramp_up_method
        self.wmax = tp.max_unsupervised * tp.labeled_size / tp.num_train_steps
        self.alpha = tp.alpha

        # use state dict instead of named_parameters when batch norm exits
        for param, ema_param in zip(self.model.state_dict().values(), self.ema_model.state_dict().values.values()):
            param.data.copy_(ema_param.data)

    def step(self):
        self.step +=1
        self.epoch = int(self.step//self.num_train_steps)
        # alpha = min(1 - 1 /(self.step+1), self.alpha)
        for name in self.model.named_parameters():
            param = self.model.state_dict()[name]
            ema_param = self.ema_model.state_dict()[name]
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1-self.alpha))
                if self.step % self.log_step==0:
                    self.tb.add_histogram(name, param, global_step=self.step)
                    self.tb.add_histogram(name+'_ema', ema_param, global_step=self.step)

    def compute_loss(self, features, logits):
        weight = ramp_up(self.epoch, self.epoch_size, self.ramp_up_method) * self.wmax

        labels = features['label']
        cond = labels >= 0
        # supervised_loss
        self.supervised_loss = self.loss_fn(logits[cond], labels[cond])

        # consistency loss
        with torch.no_grad():
            teacher_logits = self.ema_model(features)
            self.teacher_loss = self.loss_fn(teacher_logits[cond], labels[cond])

        self.consistency_loss = torch.mean((F.softmax(logits, dim=1) - F.softmax(teacher_logits, dim=1)) ** 2)

        self.tb.add_scalars('loss/sup_loss', {
            'student': self.supervised_loss,
            'teacher': self.teacher_loss
        }, global_step=self.step)

        self.tb.add_scalar('loss/consistency', self.consistency_loss, global_step=self.step)
        self.tb.add_scalar('loss/weight', weight, global_step=self.step)
        loss = self.supervised_loss + weight * self.consistency_loss
        return loss