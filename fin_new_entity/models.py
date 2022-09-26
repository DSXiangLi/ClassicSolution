# -*-coding:utf-8 -*-
import torch
from torch import nn
from transformers import BertModel
from src.layers.crf import CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from src.loss import seqlabel_loss_wrapper


class BertSoftmax(nn.Module):
    def __init__(self, tp):
        super(BertSoftmax, self).__init__()
        self.tp = tp
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.dropout_layer = nn.Dropout(tp.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, tp.label_size)

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        sequence_output = outputs[0]
        sequence_output = self.dropout_layer(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

    def compute_loss(self, features, logits):
        loss = seqlabel_loss_wrapper(logits, features['label_ids'], features['attention_mask'], self.loss_fn)
        return loss

    def decode(self, features, logits):
        preds = torch.argmax(logits, dim=-1)
        return preds

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.tp.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(params, lr=self.tp.lr, eps=self.tp.epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=self.tp.num_train_steps,
                                                    num_warmup_steps=self.tp.warmup_steps)
        return optimizer, scheduler


class BertCrf(nn.Module):
    def __init__(self, tp):
        super(BertCrf, self).__init__()
        self.tp = tp
        self.label_size = tp.label_size
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.dropout = nn.Dropout(tp.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, tp.label_size)
        self.crf = CRF(num_tags=tp.label_size, batch_first=True)

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

    def decode(self, features, logits):
        preds = self.crf.decode(emissions=logits, mask=features['attention_mask'].bool())
        return preds

    def compute_loss(self, features, logits):
        loss = -1 * self.crf(emissions=logits, tags=features['label_ids'],
                             mask=features['attention_mask'].bool(), reduction='mean')
        return loss

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.tp.lr,
             'weight_decay': self.tp.weight_decay},
            {'params': [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.tp.lr,
             'weight_decay': 0.0},
            {'params': self.classifier.parameters(), 'lr': self.tp.lr},
            {'params': self.crf.parameters(), 'lr': self.tp.crf_lr}
        ]

        optimizer = AdamW(params, lr=self.tp.lr, eps=self.tp.epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=self.tp.num_train_steps,
                                                    num_warmup_steps=self.tp.warmup_steps)
        return optimizer, scheduler

