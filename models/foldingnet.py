import os, sys
import torch
import torch.nn as nn

import numpy as np


class FoldingNet(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.in_channel = in_channel

        a = torch.linspace(-1., 1., steps=32, dtype=torch.float).view(1, 32).expand(32, 32).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=32, dtype=torch.float).view(32, 1).expand(32, 32).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

    def forward(self, x):
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, 1024)
        seed = self.folding_seed.view(1, 2, 1024).expand(bs, 2, 1024).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class NormalNet(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        self.in_channel = in_channel

        self.layers = nn.Sequential(
            nn.Conv1d(in_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

    def forward(self, features, xyz):
        bs = features.size(0)
        features = features.view(bs, self.in_channel, 1).expand(bs, self.in_channel, 1024)
        x = torch.cat([xyz.transpose(1, 2), features], dim=1)
        x = self.layers(x)
        return x
