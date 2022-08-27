# -*-coding:utf-8 -*-
from torch import nn


class Kmax_Pooling(nn.Module):
    def __init__(self, k):
        super(Kmax_Pooling, self).__init__()
        self.k = k

    def forward(self, x, dim):
        index = x.topk(self.k, dim=dim)[1]  # return topk index in descending order
        index = index.sort(dim=dim)[0]  # return topk index in ascending order
        return x.gather(dim, index)
