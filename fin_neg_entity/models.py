# -*-coding:utf-8 -*-

from torch import nn
from transformers import BertModel


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