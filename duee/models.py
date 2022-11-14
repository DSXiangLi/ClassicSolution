# -*-coding:utf-8 -*-
from torch import nn
import torch
from src.models.bert import BertClassifier
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup


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


class BertHierClassifier(BertClassifier):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, tp):
        super(BertHierClassifier, self).__init__(tp)
        self.loss_fn = HierarchyLoss(tp.hier_relation, tp.hier_weight, tp.loss_fn)

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'], self.classifier.weight)
        return loss


class BertSlotClassifier(nn.Module):
    """Bert Model for Classification Tasks. with certain slots embedding enhancement
    """

    def __init__(self, tp):
        super(BertSlotClassifier, self).__init__()
        self.tp = tp
        self.label_size = tp.label_size
        self.loss_fn = tp.loss_fn
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.dropout_layer = nn.Dropout(tp.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1) ## binary logits

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        sequence_output = outputs[0]
        slot_output = sequence_output[:, self.tp.slot_positions,: ]
        batch_size, label_size, emb_size = slot_output.shape
        slot_output = slot_output.reshape(batch_size * label_size, emb_size)
        logits = self.classifier(slot_output) # batch* label_sizse * 1
        logits = logits.squeeze()
        logits = logits.reshape(batch_size, label_size)
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
