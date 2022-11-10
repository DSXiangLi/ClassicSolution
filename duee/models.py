# -*-coding:utf-8 -*-
import torch

def hierarchy_regularize_wrapper(hier_map, label2idx, loss_fn, device):
    """ Only support hierarchical text classification with BCELoss
    references: http://www.cse.ust.hk/~yqsong/papers/2018-WWW-Text-GraphCNN.pdf
                http://www.cs.cmu.edu/~sgopal1/papers/KDD13.pdf
    """
    def func(logits, labels, label_emb):
        reg_loss = 0
        for children in hier_map.values():
            if not children:
                # 没有子类的父类
                continue
            # 也可以直接存储id，这里为了方便debug存储的是label
            children_ids = torch.tensor([label2idx[i] for i in children], dtype=torch.long).to(device)
            children_emb = torch.index_select(label_emb, dim=0, index=children_ids).to(device)