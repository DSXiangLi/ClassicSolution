# -*-coding:utf-8 -*-
import torch
from torch import nn


class Textrcnn(nn.Module):
    """
    Bilstm/GRU stack CNN Layer
    """
    def __init__(self, tp):
        super(Textrcnn, self).__init__()
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

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=tp.embedding_dim + tp.hidden_size * 2,
                      out_channels=tp.filter_size,
                      kernel_size=tp.kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(tp.max_seq_len - tp.kernel_size * 2 + 2))
        )
        self.dropout = nn.Dropout(tp.dropout_rate)
        self.fc = nn.Linear(tp.filter_size, tp.label_size)

    def forward(self, features):
        x = self.embedding(features['token_ids'])  # (batch_size, seq_len, emb_dim)

        # 1. lstm/gru layer
        rnn_x = self.rnn(x)[0]  # output: (batch_size, seq_len, hidden_size [*2 if bilstm])

        # 2. concat origianl input and bidirectional hidden output
        x = torch.cat([x, rnn_x], dim=2)  # output （batch_size, seq_len, emb_dim + hidden_size）

        # 3. Convolution
        x = self.conv(x.permute(0, 2, 1)).squeeze(-1)  # (batch_size, filter_size)
        x = self.dropout(x)

        logits = self.fc(x)
        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss