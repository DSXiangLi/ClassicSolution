# -*-coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
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



class BertSpan(nn.Module):
    def __init__(self, tp):
        super(BertSpan, self).__init__()
        self.tp = tp
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.dropout_layer = nn.Dropout(tp.dropout_rate)
        self.start_classifier = nn.Linear(self.bert.config.hidden_size, tp.label_size)
        # end classifier : use start output and bert output
        self.end_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size+tp.label_size, self.tp.hidden_size),
            nn.Tanh(),
            nn.LayerNorm(self.tp.hidden_size),
            nn.Linear(self.tp.hidden_size, self.tp.label_size)
        )

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids1, label_ids2}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        sequence_output = outputs[0]
        sequence_output = self.dropout_layer(sequence_output)
        logits_start = self.start_classifier(sequence_output)
        start_pos = features.get('label_start')

        if start_pos is not None and self.training:
            # Training Stage: use input pos as soft label
            batch_size, seq_len = start_pos.shape
            start_soft_label = torch.zeros(batch_size, seq_len, self.tp.label_size, device=features['input_ids'].device)
            start_soft_label.scatter_(2, start_pos.unsqueeze(2),1)
        else:
            # Inference Stage: use prediction as soft label
            start_soft_label = F.softmax(logits_start, dim=-1)
        logits_end = self.end_classifier(torch.concat([sequence_output, start_soft_label], dim=-1))
        return logits_start, logits_end

    def compute_loss(self, features, logits_pair):
        loss1 = seqlabel_loss_wrapper(logits_pair[0], features['label_start'], features['attention_mask'], self.loss_fn)
        loss2 = seqlabel_loss_wrapper(logits_pair[1], features['label_end'], features['attention_mask'], self.loss_fn)
        loss = loss1+loss2
        return loss

    def decode(self, features, logits_pair):
        start_pred = torch.argmax(logits_pair[0], dim=-1)
        end_pred = torch.argmax(logits_pair[1], dim=-1)
        return start_pred, end_pred

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