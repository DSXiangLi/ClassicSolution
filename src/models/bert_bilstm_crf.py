# -*-coding:utf-8 -*-
import torch
from torch import nn
from transformers import BertModel
from src.layers.crf import CRF
from transformers import AdamW, get_linear_schedule_with_warmup


class BertBilstmCrf(nn.Module):
    def __init__(self, tp):
        super(BertBilstmCrf, self).__init__()
        self.tp = tp
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, tp.hidden_size,
                            num_layers=self.tp.num_layers,
                            bidirectional=True, dropout=tp.dropout_rate, batch_first=True)
        self.classifier = nn.Linear(int(tp.hidden_size * 2), tp.label_size)
        self.crf = CRF(num_tags=tp.label_size, batch_first=True)

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        lstm_output = self.lstm(outputs[0])
        logits = self.classifier(lstm_output[0])
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
            {'params': self.lstm.parameters(), 'lr': self.tp.lr},
            {'params': self.crf.parameters(), 'lr': self.tp.crf_lr}
        ]

        optimizer = AdamW(params, lr=self.tp.lr, eps=self.tp.epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=self.tp.num_train_steps,
                                                    num_warmup_steps=self.tp.warmup_steps)
        return optimizer, scheduler