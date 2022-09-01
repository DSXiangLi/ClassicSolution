# -*-coding:utf-8 -*-
"""
    Refactor models for word enhance input
"""
import torch
import torch.nn as nn

from src.enhancement.mixup import mixup
from src.models.kmax_pool import Kmax_Pooling
from src.enhancement.consistency import TemporalEnsemble, MeanTeacher
from src.enhancement.min_entropy import PseudoLabel


class Textcnn(nn.Module):
    def __init__(self, tp):
        super(Textcnn, self).__init__()
        self.tp = tp
        self.loss_fn = tp.loss_fn
        self.embedding1 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding1, dtype=torch.float32), freeze=False)
        self.embedding2 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding2, dtype=torch.float32), freeze=False)
        self.label_size = tp.label_size
        self.projector = nn.Sequential(
            nn.Linear(tp.embedding_dim, tp.hidden_size),
            nn.BatchNorm1d(tp.hidden_size),  # order of BN and Relu is case relevant
            nn.ReLU(True)
        )
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=tp.hidden_size,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(tp.max_seq_len - kernel_size * 2 + 2))
        )
            for kernel_size in tp.kernel_size_list])
        self.dropout = nn.Dropout(tp.dropout_rate)
        self.fc = nn.Linear(int(tp.filter_size * len(tp.kernel_size_list)), tp.label_size)

    def forward(self, features):
        x1 = self.embedding1(features['token_ids'])  # (batch_size, seq_len, emb_dim)
        x2 = self.embedding2(features['word_ids'])
        x = torch.cat([x1, x2], dim=-1)
        # BatchNorm1d is applied on (N,C,L)C or (N,L)L
        x = self.projector(x.contiguous().view(-1, x.size(-1))).view(x.size(0), x.size(1),
                                                                     -1)  # (batch_size, seq_len, hidden_size)
        x = [conv(x.permute(0, 2, 1)).squeeze(-1) for conv in self.convs]  # input (batch_size, channel, # seq_len)
        x = torch.cat(x, dim=1)  # (batch_size, sum(filter_size))
        x = self.dropout(x)

        logits = self.fc(x)
        return logits

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.tp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.tp.scheduler_params)
        return optimizer, scheduler

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss


class TextcnnMixup(nn.Module):
    def __init__(self, tp):
        super(TextcnnMixup, self).__init__()
        self.tp = tp
        self.embedding1 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding1, dtype=torch.float32), freeze=False)
        self.embedding2 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding2, dtype=torch.float32), freeze=False)
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size
        self.alpha = tp.mixup_alpha
        self.projector = nn.Sequential(
            nn.Linear(tp.embedding_dim, tp.hidden_size),
            nn.BatchNorm1d(tp.hidden_size),
            nn.ReLU(True)
        )
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=tp.hidden_size,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(tp.max_seq_len - kernel_size * 2 + 2))
        )
            for kernel_size in tp.kernel_size_list])
        self.dropout = nn.Dropout(tp.dropout_rate)
        self.fc = nn.Linear(int(tp.filter_size * len(tp.kernel_size_list)), tp.label_size)

    def forward(self, features):
        x1 = self.embedding1(features['token_ids'])  # (batch_size, seq_len, emb_dim)
        x2 = self.embedding2(features['word_ids'])
        x = torch.cat([x1, x2], dim=-1)

        # BatchNorm1d is applied on (N,C,L)C or (N,L)L
        if features.get('label') is not None:
            if self.training:
                x, self.ymix = mixup(x, features['label'], self.label_size, self.alpha)
            else:
                # don't do mixup in eval mode
                self.ymix = features['label']

        x = self.projector(x.contiguous().view(-1, x.size(-1))).view(x.size(0), x.size(1),
                                                                     -1)  # (batch_size, seq_len, hidden_size)

        x = [conv(x.permute(0, 2, 1)).squeeze(-1) for conv in self.convs]  # input (batch_size, channel, # seq_len)
        x = torch.cat(x, dim=1)  # (batch_size, sum(filter_size))
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, self.ymix)
        return loss

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.tp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.tp.scheduler_params)
        return optimizer, scheduler


class Fasttext(nn.Module):
    """
    Fasttext: average word embedding
    """

    def __init__(self, tp):
        super(Fasttext, self).__init__()
        self.tp = tp
        self.embedding1 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding1, dtype=torch.float32), freeze=False)
        self.embedding2 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding2, dtype=torch.float32), freeze=False)
        self.projector = nn.Sequential(
            nn.Linear(tp.embedding_dim, tp.hidden_size),
            nn.BatchNorm1d(tp.hidden_size),
            nn.ReLU(True)
        )
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size
        self.fc = nn.Linear(tp.hidden_size, tp.label_size)

    def forward(self, features):
        x1 = self.embedding1(features['token_ids'])  # (batch_size, seq_len, emb_dim)
        x2 = self.embedding2(features['word_ids'])
        x = torch.cat([x1, x2], dim=-1)
        # project the input embedding
        x = self.projector(x.contiguous().view(-1, x.size(-1))).view(x.size(0), x.size(1),
                                                                     -1)  # (batch_size, seq_len, hidden_size)

        # Average Embedding: ignore the padding part
        x = torch.sum(x, dim=1) / torch.sum(features['attention_mask'], dim=1).unsqueeze(1)  # (batch_size, hidden_size)

        logits = self.fc(x)
        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss

    def get_optimizer(self):
        if self.tp.diff_lr:
            params = [
                {'params': self.embedding1.parameters(), 'lr': 1e-4},
                {'params': self.embedding2.parameters(), 'lr': 1e-4},
                {'params': self.projector.parameters(), 'lr': 1e-4},
                {'params': self.fc.parameters()},
            ]
            optimizer = torch.optim.Adam(params, lr=self.tp.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.tp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.tp.scheduler_params)
        return optimizer, scheduler


class Textrnn(nn.Module):
    """
        1 layer RNN[LSTM/GRU], output last hidden state for prediction
    """

    def __init__(self, tp):
        super(Textrnn, self).__init__()
        self.tp = tp
        self.embedding1 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding1, dtype=torch.float32), freeze=False)
        self.embedding2 = nn.Embedding.from_pretrained(torch.tensor(tp.embedding2, dtype=torch.float32), freeze=False)
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size
        self.projector = nn.Sequential(
            nn.Linear(tp.embedding_dim, tp.hidden_size),
            nn.BatchNorm1d(tp.hidden_size),  # order of BN and Relu is case relevant
            nn.ReLU(True)
        )
        if tp.layer_type == 'lstm':
            self.rnn = nn.LSTM(input_size=tp.hidden_size,
                               hidden_size=tp.hidden_size,
                               num_layers=tp.num_layers,
                               bias=True,
                               batch_first=True,
                               bidirectional=True
                               )
        elif tp.layer_type == 'gru':
            self.rnn = nn.GRU(input_size=tp.hidden_size,
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
        x1 = self.embedding1(features['token_ids'])  # (batch_size, seq_len, emb_dim)
        x2 = self.embedding2(features['word_ids'])
        x = torch.cat([x1, x2], dim=-1)
        x = self.projector(x.contiguous().view(-1, x.size(-1))).view(x.size(0), x.size(1),
                                                                     -1)  # (batch_size, seq_len, hidden_size)
        x = self.rnn(x)[0]  # output: (batch_size, seq_len, hidden_size [*2 if bilstm])
        x = self.kmax_pool(x, dim=1)  # topk feature on seq_len axis
        x = x.view(x.size(0), -1)  # flatten

        x = self.dropout(x)
        logits = self.fc(x)
        return logits

    def compute_loss(self, features, logits):
        loss = self.loss_fn(logits, features['label'])
        return loss

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.tp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.tp.scheduler_params)
        return optimizer, scheduler


class TextcnnTemporal(TemporalEnsemble, Textcnn):
    def __init__(self, tp):
        Textcnn.__init__(self, tp)
        TemporalEnsemble.__init__(self, tp)


class TextcnnPseudoLabel(PseudoLabel, Textcnn):
    def __init__(self, tp):
        # init nn.Module before others
        Textcnn.__init__(self, tp)
        PseudoLabel.__init__(self, tp)


class TextcnnMeanTeacher(MeanTeacher, Textcnn):
    def __init__(self, tp, tb):
        Textcnn.__init__(self, tp)
        MeanTeacher.__init__(self, tp, tb)