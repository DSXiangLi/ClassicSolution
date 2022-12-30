# -*-coding:utf-8 -*-
"""
    Noisy Robust Loss
    - Generalize Cross entropy
    - Symmetric Cross Entropy
    - Bootstrap Cross Entropy
    - Peer Loss

    Class Imbalance Loss
    - Focal Loss

    Multi Label
    - MultiLabel Cross Entropy
"""
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


def seqlabel_loss_wrapper(logits, label_ids, attention_mask, loss_func):
    # remove padding and CLS, SEP from loss calcualtion
    mask = torch.logical_and(attention_mask.view(-1) == 1, label_ids.view(-1) >= 0)
    # flatten logits: [batch, seq_len, label_size] -> [batch * seq_len, label_size]
    label_size = logits.size(dim=-1)
    logits = logits.view(-1, label_size)[mask]
    label_ids = label_ids.view(-1)[mask]
    loss = loss_func(logits, label_ids)
    return loss


class GeneralizeCrossEntropy(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizeCrossEntropy, self).__init__()
        self.q = q

    def forward(self, logits, labels):
        # Negative box cox: (1-f(x)^q)/q
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        loss = (1 - torch.pow(torch.sum(labels * probs, dim=-1), self.q) )/ self.q
        loss = torch.mean(loss)
        return loss


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-10

    def forward(self, logits, labels):
        # KL(p|q) + KL(q|p)
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        # KL
        y_true = torch.clip(labels, self.eps, 1.0 - self.eps)
        y_pred = probs
        ce = -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=-1))

        # reverse KL
        y_true = probs
        y_pred = torch.clip(labels, self.eps, 1.0 - self.eps)
        rce = -torch.mean(torch.sum(y_true * torch.log(y_pred), dim=-1))

        return self.alpha * ce + self.beta * rce


class BootstrapCrossEntropy(nn.Module):
    def __init__(self, beta=0.95, is_hard=0):
        super(BootstrapCrossEntropy, self).__init__()
        self.beta = beta
        self.is_hard = is_hard

    def forward(self, logits, labels):
        # (beta * y + (1-beta) * p) * log(p)
        labels = F.one_hot(labels, num_classes=logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        probs = torch.clip(probs, self.eps, 1 - self.eps)

        if self.is_hard:
            pred_label = F.one_hot(torch.argmax(probs, dim=-1), num_classes=logits.shape[-1])
        else:
            pred_label = probs
        loss = torch.sum((self.beta * labels + (1 - self.beta) * pred_label) * torch.log(probs), dim=-1)
        loss = torch.mean(- loss)
        return loss


class PeerLoss(nn.Module):
    def __init__(self, loss, alpha=0.5):
        super(PeerLoss, self).__init__()
        self.alpha = alpha
        self.loss = loss

    def forward(self, logits, labels):
        index = list(range(labels.shape[0]))
        rand_index = random.shuffle(index)
        rand_labels = labels[rand_index]
        loss_true = self.loss(logits, labels)
        loss_rand = self.loss(logits, rand_labels)
        loss = loss_true - self.alpha * loss_rand
        return loss


class BinaryFocal(nn.Module):
    def __init__(self, gamma=2, class_weight=None):
        super(BinaryFocal, self).__init__()
        self.gamma = gamma
        self.class_weight = class_weight

    def forward(self, logits, labels):
        # y * log(p) * (1-p) *** r
        assert len(self.class_weight) == logits.shape[-1], 'class wight len should be equal to label size'
        labels = F.one_hot(labels, num_classes=logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        loss = - torch.sum(labels * torch.log(probs) * torch.pow(1-probs, self.gamma), dim=-1)
        imbalance = torch.sum(labels * self.class_weight)
        loss = torch.mean(loss * imbalance)
        return loss


class MultiLabelFocal(nn.Module):
    def __init__(self, gamma=2):
        #针对多标签问题，默认不适用类权重
        super(MultiLabelFocal, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        # y * log(p) * (1-p) *** r
        probs = F.sigmoid(logits)
        loss = - torch.sum(labels * torch.log(probs) * torch.pow(1 - probs, self.gamma), dim=-1)
        loss = torch.mean(loss)
        return loss



class MultilabelCrossEntropy(nn.Module):
    def __init__(self):
        super(MultilabelCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        """
        多标签分类的交叉熵
        https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * labels) * logits  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - logits * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - logits) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()
