# -*-coding:utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Mixup(nn.Module):
    def __init__(self, label_size, alpha):
        super(Mixup, self).__init__()
        self.label_size = label_size
        self.alpha = alpha

    def forward(self, input_x, input_y):
        """
        Do in Module, so that training mode will be used in inference
        """
        if not self.training:
            return input_x, input_y
        batch_size = input_x.size()[0]
        input_y = F.one_hot(input_y, num_classes=self.label_size)

        # get mix ratio
        mix = np.random.beta(self.alpha, self.alpha)
        mix = np.max([mix, 1 - mix])

        # get random shuffle sample
        index = torch.randperm(batch_size)
        random_x = input_x[index, :]
        random_y = input_y[index, :]

        xmix = input_x * mix + random_x * (1 - mix)
        ymix = input_y * mix + random_y * (1 - mix)
        return xmix, ymix