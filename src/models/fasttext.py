# -*-coding:utf-8 -*-
import torch
from torch import nn


class Fasttext(nn.Module):
    """
    Fasttext: average word embedding
    """

    def __init__(self, tp):
        super(Fasttext, self).__init__()
        if tp.get('embedding') is None:
            self.embedding = nn.Embedding(tp.vocab_size, tp.embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(tp.embedding, dtype=torch.float32), freeze=False)
        self.projector = nn.Sequential(
            nn.Linear(tp.embedding_dim, tp.hidden_size),
            nn.BatchNorm1d(tp.hidden_size),
            nn.ReLU(True)
        )
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size
        self.fc = nn.Linear(tp.hidden_size, tp.label_size)

    def forward(self, features):
        x = self.embedding(features['token_ids'])  # (batch_size, seq_len, emb_dim)

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
