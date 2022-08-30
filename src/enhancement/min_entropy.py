# -*-coding:utf-8 -*-
import torch
import torch.nn.functional as F


class MinEnt(object):
    def __init__(self, tp):
        self.loss_fn = tp.loss_fn
        self.alpha = tp.alpha
        self.supervised_loss = 0
        self.unsupervised_loss = 0

    def compute_loss(self, features, logits):
        labels = features['label']
        cond = labels >= 0
        # supervised_loss
        self.supervised_loss = self.loss_fn(logits[cond], labels[cond])

        # unsupervised entropy
        cond = labels <0
        probs = F.softmax(logits[cond], dim=-1)
        self.unsupervised_loss = torch.sum(probs * torch.log(probs), dim=-1)
        loss = self.supervised_loss + self.alpha * self.unsupervised_loss
        return loss


class PseudoLabel(object):
    def __init__(self, tp ):
        self.T1 = tp.T1
        self.T2 = tp.T2
        self.alpha_f = tp.alpha_f
        self.cur_epoch = 0
        self.loss_fn = tp.loss_fn
        self.alpha_t = 0
        self.supervised_loss = 0
        self.unsupervised_loss = 0

    def compute_loss(self, features, logits):
        labels = features['label']
        cond = labels >= 0
        # supervised_loss
        self.supervised_loss = self.loss_fn(logits[cond], labels[cond])

        # unsupervised_loss
        with torch.no_grad():
            pseudo_label = torch.argmax(logits, dim=-1)
        cond = labels < 0
        self.unsupervised_loss = self.loss_fn(logits[cond], pseudo_label[cond])

        loss = self.supervised_loss + self.alpha_t * self.unsupervised_loss

        return loss

    def epoch_update(self):
        self.cur_epoch += 1
        self.alpha_t = self.unlabeled_weight()

    def unlabeled_weight(self):
        alpha = 0.0
        if self.cur_epoch > self.T1:
            alpha = (self.cur_epoch - self.T1) / (self.T2 - self.T1) * self.alpha_f
            if self.cur_epoch > self.T2:
                alpha = self.alpha_f
        return alpha


