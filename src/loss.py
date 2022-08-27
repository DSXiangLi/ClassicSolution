# -*-coding:utf-8 -*-
import torch

def seqlabel_loss_wrapper(logits, label_ids, attention_mask, loss_func):
    # remove padding and CLS, SEP from loss calcualtion
    mask = torch.logical_and(attention_mask.view(-1) == 1, label_ids.view(-1)>=0)
    # flatten logits: [batch, seq_len, label_size] -> [batch * seq_len, label_size]
    label_size = logits.size(dim=-1)
    logits = logits.view(-1, label_size)[mask]
    label_ids = label_ids.view(-1)[mask]
    loss = loss_func(logits, label_ids)
    return loss