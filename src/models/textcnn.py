# -*-coding:utf-8 -*-
import torch
from torch import nn


class Textcnn(nn.Module):
    def __init__(self, tp):
        super(Textcnn, self).__init__()
        if tp.get('embedding') is None:
            self.embedding = nn.Embedding(tp.vocab_size, tp.embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(tp.embedding, dtype=torch.float32),freeze=False)
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=tp.embedding_dim,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(tp.max_seq_len - kernel_size * 2 + 2))
        )
            for kernel_size in tp.kernel_size_list])
        self.dropout = nn.Dropout(tp.dropout_rate)
        self.fc = nn.Linear(int(tp.filter_size * len(tp.kernel_size_list)), tp.label_size)

    def forward(self, features):
        x = self.embedding(features['token_ids'])  # (batch_size, seq_len, emb_dim)

        x = [conv(x.permute(0, 2, 1)).squeeze(-1) for conv in self.convs]  # input (batch_size, channel, # seq_len)
        x = torch.cat(x, dim=1)  # (batch_size, sum(filter_size))
        x = self.dropout(x)

        logits = self.fc(x)
        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss


class Textcnn2(nn.Module):
    """
    2层CNN + 2层FC
    """

    def __init__(self, tp):
        super(Textcnn2, self).__init__()
        if tp.get('embedding') is None:
            self.embedding = nn.Embedding(tp.vocab_size, tp.embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(tp.embedding, dtype=torch.float32), freeze=False)
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=tp.embedding_dim,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(tp.filter_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=tp.filter_size,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(tp.filter_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(tp.max_seq_len - tp.kernel_size * 2 + 2))
        )
            for kernel_size in tp.kernel_size_list])

        self.dropout = nn.Dropout(tp.dropout_rate)

        self.fc = nn.Sequential(
            nn.Linear(int(tp.filter_size * len(tp.kernel_size_list)), tp.hidden_size),
            nn.BatchNorm1d(tp.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(tp.hidden_size, tp.label_size)
        )

    def forward(self, features):
        emb = self.embedding(features['token_ids'])  # (batch_size, seq_len, emb_dim)
        emb = [conv(emb.permute(0, 2, 1)).squeeze(-1) for conv in
               self.convs]  # Conv1d input shape (batch_size, channel, seq_len)
        x = torch.cat(emb, dim=1)  # (batch_size, sum(filter_size))
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss