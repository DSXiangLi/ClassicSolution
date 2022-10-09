# -*-coding:utf-8 -*-
"""
    Reference: 苏神 https://kexue.fm/archives/8373/
"""
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from src.layers.position_embedding import RoPE


class GlobalPointer(nn.Module):
    def __init__(self, input_size, num_head, head_size, loss_fn):
        super().__init__()
        self.num_head = num_head
        self.head_size = head_size
        self.projector = nn.Linear(input_size, num_head * head_size * 2)
        self.rope = RoPE(output_dim=head_size)
        self.loss_fn = loss_fn

    def forward(self, input_):
        batch_size, seq_len, *args = input_.shape
        # project query and key for each head
        hidden = self.projector(input_)
        # [batch_size, seq_len, num_head, head_size*2]
        hidden = torch.stack(torch.split(hidden, self.head_size * 2, dim=-1), dim=-2)
        # split query and key: [batch, seq_len, num_head, head_size]
        qw, kw = hidden[..., :self.head_size], hidden[..., self.head_size:]
        # apply rotary encoding, shape unchanged
        qw = self.rope(qw)
        kw = self.rope(kw)

        # Binary logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        logits = logits / self.num_head ** 0.5  # scale logits
        return logits

    def get_mask(self, features, logits):
        # seq_len -> head * seq_len * seq_Len
        mask = features['attention_mask']
        batch_size, seq_len = mask.shape
        ## mask padding: expand saves memory usage compared to repeate
        mask1 = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        # add head dim
        if self.num_head == 1:
            mask1 = mask1.unsqueeze(-3)
        else:
            mask1 = mask1.expand(batch_size, self.num_head, seq_len, seq_len)
        ## mask lower triangle
        mask2 = torch.triu(torch.ones_like(logits), diagonal=-1)
        mask = torch.logical_and(mask1 == 1, mask2 == 1)
        return mask

    def compute_loss(self, features, logits):
        mask = self.get_mask(features, logits)
        ## reshape to 2dim and remove the padding and lower triangle
        batch_size, *args = logits.shape
        logits = logits.reshape(batch_size * self.num_head, -1)
        labels = features['label_ids'].reshape(batch_size * self.num_head, -1)
        mask = mask.reshape(batch_size * self.num_head, -1)
        logits = logits[mask]
        labels = labels[mask]
        loss = self.loss_fn(logits, labels.float())
        return loss

    def decode(self, features, logits):
        preds = (logits > 0).long()
        return preds


class BertGlobalPointer(nn.Module):
    def __init__(self, tp):
        super(BertGlobalPointer, self).__init__()
        self.tp = tp
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.global_pointer = GlobalPointer(self.bert.config.hidden_size, tp.num_head, tp.head_size, tp.loss_fn)

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        logits = self.global_pointer(outputs[0])
        return logits

    def decode(self, features, logits):
        return self.global_pointer.decode(features, logits)

    def compute_loss(self, features, logits):
        return self.global_pointer.compute_loss(features, logits)

    def get_mask(self, features, logits):
        return self.global_pointer.get_mask(features, logits)

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.tp.lr,
             'weight_decay': self.tp.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.tp.lr,
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(params, lr=self.tp.lr, eps=self.tp.epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=self.tp.num_train_steps,
                                                    num_warmup_steps=self.tp.warmup_steps)
        return optimizer, scheduler