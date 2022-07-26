# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : transform.py
# Author     ：Wang Yuhao
# Description：
"""
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

no_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
