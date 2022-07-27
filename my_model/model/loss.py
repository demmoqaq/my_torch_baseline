# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loss.py
# Author     ：Wang Yuhao
# Description：
"""
import torch
from torch.nn import CrossEntropyLoss


class ComputeLoss:
    def __init__(self):
        self.MSELoss = CrossEntropyLoss(reduction="none")

    def __call__(self, outputs, targets):
        loss = self.MSELoss(outputs, targets)
        total_loss = loss
        return total_loss, torch.cat((loss)).detach()
