# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model_name.py
# Author     ：Wang Yuhao
# Description：
"""
from torch import nn


class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 6, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_part = nn.Sequential(
            nn.Flatten(1, 3),
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv_part(x)
        x = self.fc_part(x)
        return x
