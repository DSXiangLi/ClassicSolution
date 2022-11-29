# -*-coding:utf-8 -*-
import torch
from torch import nn


class HierarchyLoss(nn.Module):
    def __init__(self, hier_relation, hier_weight, loss_fn):
        super(HierarchyLoss, self).__init__()
        self.hier_relation = hier_relation
        self.hier_weight = hier_weight
        self.loss_fn = loss_fn

    def forward(self, logits, labels, label_emb):
        device = logits.device
        recursive_loss = 0
        for idx in range(len(label_emb)):
            if idx not in self.hier_relation:
                continue
            if len(self.hier_relation[idx]) == 0:
                continue
            children_ids = self.hier_relation[idx]
            children_ids = torch.tensor(children_ids, dtype=torch.long).to(device)
            children_emb = torch.index_select(label_emb, dim=0, index=children_ids).to(device)
            parent_emb = torch.index_select(label_emb, dim=0, index=torch.tensor(idx, device=device)).to(device)
            parent_emb = parent_emb.repeat(children_ids.size()[0], 1)
            diff = parent_emb - children_emb
            recursive_loss += torch.norm(diff, p=2) ** 2
        loss = self.loss_fn(logits, labels) + self.hier_weight * recursive_loss
        return loss
