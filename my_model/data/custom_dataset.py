# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : list_image_dataset.py
# Author     ：Wang Yuhao
# Description：
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, args, transform, target_transform=None):
        """:args should contain img_dir: str, annotations_file_dir: str, in_memory: bool"""
        super(CustomImageDataset, self).__init__()
        self.img_labels = pd.read_csv(args.annotations_file_dir)
        self.img_dir = args.img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        # directly load data into memory (for faster training)
        if args.in_memory:
            for img_name in self.img_labels.iloc[:, 0].values():
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path)
                image = self.transform(image)
                self.images.append(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if not self.images:  # image not in memory
            image = Image.open(img_path)
            image = self.transform(image)
        else:  # already in memory
            image = self.images[idx]

        label = self.img_labels.iloc[idx, 1]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomTableDataset(Dataset):  # todo: unfinish
    def __init__(self, args, transform, target_transform=None):
        """:args should contain img_dir: str, annotations_file_dir: str, in_memory: bool"""
        super(CustomTableDataset, self).__init__()
        self.img_labels = pd.read_csv(args.annotations_file_dir)
        self.img_dir = args.img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        # directly load data into memory (for faster training)
        if args.in_memory:
            for img_name in self.img_labels.iloc[:, 0].values():
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path)
                image = self.transform(image)
                self.images.append(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if not self.images:  # image not in memory
            image = Image.open(img_path)
            image = self.transform(image)
        else:  # already in memory
            image = self.images[idx]

        label = self.img_labels.iloc[idx, 1]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label