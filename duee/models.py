# -*-coding:utf-8 -*-
import torch


def hierarchy_loss_wrapper(hier_relation, hier_weight, loss_fn , device):
    def func(logits, labels, label_emb):
        # label_emb : label_size * emb_size
        recursive_loss = 0
        for idx in range(len(label_emb)):
            if idx not in hier_relation:
                continue
            if len(hier_relation[idx])==0:
                continue
            children_ids = hier_relation[idx]
            children_ids = torch.tensor(children_ids,  dtype=torch.long).to(device)
            children_emb = torch.index_select(label_emb, dim=0, index=children_ids).to(device)
            parent_emb = torch.index_select(label_emb, dim=0, index=torch.tensor(idx)).to(device)
            parent_emb = parent_emb.repeat(children_ids.size()[0],1)
            diff = parent_emb - children_emb
            recursive_loss += torch.norm(diff, p =2)**2
        loss = loss_fn(logits, labels) + hier_weight * recursive_loss
        return loss
    return func
