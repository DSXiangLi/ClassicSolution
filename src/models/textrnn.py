# -*-coding:utf-8 -*-
from torch import nn
import torch

from src.models.kmax_pool import Kmax_Pooling


class Textrnn(nn.Module):
    """
        1 layer RNN[LSTM/GRU], output last hidden state for prediction
    """
    def __init__(self, tp):
        super(Textrnn, self).__init__()
        if tp.get('embedding') is None:
            self.embedding = nn.Embedding(tp.vocab_size, tp.embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(tp.embedding, dtype=torch.float32), freeze=False)
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size

        if tp.layer_type == 'lstm':
            self.rnn = nn.LSTM(input_size=tp.embedding_dim,
                               hidden_size=tp.hidden_size,
                               num_layers=tp.num_layers,
                               bias=True,
                               batch_first=True,
                               bidirectional=True
                               )
        elif tp.layer_type == 'gru':
            self.rnn = nn.GRU(input_size=tp.embedding_dim,
                              hidden_size=tp.hidden_size,
                              num_layers=tp.num_layers,
                              bias=True,
                              bidirectional=True,
                              batch_first=True
                              )
        self.kmax_pool = Kmax_Pooling(tp.topk)
        self.dropout = nn.Dropout(tp.dropout_rate)
        self.fc = nn.Linear(int(tp.hidden_size * 2 * tp.topk), tp.label_size)

    def forward(self, features):
        x = self.embedding(features['token_ids'])  # (batch_size, seq_len, emb_dim)

        x = self.rnn(x)[0]  # output: (batch_size, seq_len, hidden_size [*2 if bilstm])
        x = self.kmax_pool(x, dim=1)  # topk feature on seq_len axis
        x = x.view(x.size(0), -1)  # flatten

        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss
