# -*-coding:utf-8 -*-
import torch
import numpy as np


def mixup(input_x, input_y, label_size, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size = input_x.size()[0]
    input_y = torch.nn.functional.one_hot(input_y, num_classes=label_size)

    # get mix ratio
    mix = np.random.beta(alpha, alpha)
    mix = np.max([mix, 1-mix])

    #get random shuffle sample
    index = torch.randperm(batch_size)
    random_x = input_x[index,:]
    random_y = input_y[index,:]

    xmix = input_x * mix + random_x * (1-mix)
    ymix = input_y * mix + random_y * (1-mix)
    return xmix, ymix
