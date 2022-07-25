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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

no_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
