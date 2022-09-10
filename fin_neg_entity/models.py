# -*-coding:utf-8 -*-

from torch import nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import init
import torch


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, tp):
        super(BertClassifier, self).__init__()
        self.tp = tp
        self.label_size = tp.label_size
        self.loss_fn = tp.loss_fn
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.dropout_layer = nn.Dropout(tp.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_size)


    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        pooled_output = self.dropout_layer(outputs.pooler_output)

        logits = self.classifier(pooled_output)

        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss

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

    class BertClassifier(nn.Module):
        """Bert Model for Classification Tasks.
        """

        def __init__(self, tp):
            super(BertClassifier, self).__init__()
            self.tp = tp
            self.label_size = tp.label_size
            self.loss_fn = tp.loss_fn
            self.bert = BertModel.from_pretrained(tp.pretrain_model)
            self.dropout_layer = nn.Dropout(tp.dropout_rate)
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_size)

        def forward(self, features):
            """
            features: {input_ids, token_type_ids, attention_mask, label_ids}
            """
            outputs = self.bert(input_ids=features['input_ids'],
                                token_type_ids=features['token_type_ids'],
                                attention_mask=features['attention_mask'])
            pooled_output = self.dropout_layer(outputs.pooler_output)

            logits = self.classifier(pooled_output)

            return logits

        def compute_loss(self, features, logits):
            loss = self.loss_fn(logits, features['label'])
            return loss

        def get_optimizer(self):
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            params = [
                {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.tp.weight_decay},
                {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = AdamW(params, lr=self.tp.lr, eps=self.tp.epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=self.tp.num_train_steps,
                                                        num_warmup_steps=self.tp.warmup_steps)
            return optimizer, scheduler



class BertMtl(nn.Module):
    """
    不对称多任务
    Task1: Sentiment Analysis
    Task2: Entity Sentiment Analysis, 使用task1的hidden和output但是不更新梯度
    """
    def __init__(self, tp):
        super(BertMtl, self).__init__()
        self.tp = tp
        self.label_size = tp.label_size
        self.loss_fn = tp.loss_fn
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.dropout_layer = nn.Dropout(tp.dropout_rate)
        # sentence sentiment task
        self.hidden_s = nn.Sequential(nn.Linear(self.bert.config.hidden_size, self.tp.hidden_s),
                                      nn.ReLU(True)
                                      )
        self.classifier_s = nn.Linear(self.tp.hidden_s, self.label_size)
        self.logits_s = None  # 辅助任务logits，计算loss不计算metrics

        # entity sentiment task
        self.hidden_e = nn.Sequential(nn.Linear(self.bert.config.hidden_size + self.label_size + self.tp.hidden_s,
                                                self.tp.hidden_e),
                                      nn.ReLU(True))
        self.classifier_e = nn.Linear(self.tp.hidden_e, self.label_size)

        ## Weight Init
        init.xavier_uniform_(self.hidden_s[0].weight)
        init.xavier_uniform_(self.hidden_e[0].weight)

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        pooled_output = self.dropout_layer(outputs.pooler_output)

        hidden_s = self.hidden_s(pooled_output)
        self.logits_s = self.classifier_s(hidden_s)

        input_e = torch.concat([pooled_output, hidden_s.data, self.logits_s.data], dim=-1)
        hidden_e = self.hidden_e(input_e)
        logits_e = self.classifier_e(hidden_e)

        return logits_e

    def compute_loss(self, features, logits):
        loss_e = self.loss_fn(logits, features['label1'])
        loss_s = self.loss_fn(self.logits_s, features['label2'])
        loss = loss_s + loss_e
        return loss

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