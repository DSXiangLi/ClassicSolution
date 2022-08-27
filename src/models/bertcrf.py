# -*-coding:utf-8 -*-
import torch
from torch import nn
from transformers import BertModel
from src.layers.crf import CRF


def pad_sequence(input_, pad_len=None, pad_value=0):
    """
    Pad List[List] sequence to same length
    """
    output = []
    for i in input_:
        output.append(i + [pad_value] * (pad_len - len(i)))
    return output


class BertCrf(nn.Module):
    def __init__(self, tp):
        super(BertCrf, self).__init__()
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
        output = (logits, )
        # CRF verbiti decode: return dynamic shape of seq_len
        preds = self.crf.decode(emissions=logits, mask=features['attention_mask'].bool())
        preds = pad_sequence(preds, pad_len=features['input_ids'].size()[-1])
        outputs += (torch.tensor(preds, device=logits.device),)
        return output

    def compute_loss(self, features, logits):
        loss = -1 * self.crf(emissions=logits, tags=features['label_ids'],
                             mask=features['attention_mask'].bool(), reduction='mean')
        return loss
