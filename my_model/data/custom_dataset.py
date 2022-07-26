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
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, annotations_file_dirs=None,
                 without_label=False, in_memory=False, read_method='pillow',
                 transform=None, target_transform=None):
        """
        :param img_dirs: list of directory names
        :param annotations_file_dirs: list of label file names
        :param in_memory:
        :param transform:
        :param target_transform:
        """
        read_methods = ['pillow', 'torchvision.io']
        assert read_method in read_methods, f'{read_method} not in required methods {read_methods}'

        super(CustomImageDataset, self).__init__()
        self.img_dir_labels = []
        if not without_label:  # normal condition for DL
            assert annotations_file_dirs is not None, 'should have annotations_file_dirs'
            for img_dir, anno_dir in zip(img_dirs, annotations_file_dirs):
                self.img_dir_labels += [[os.path.join(img_dir, info[0]), info[1]]
                                        for info in pd.read_csv(anno_dir).values.tolist()]
        else:  # without labels, in GAN or validation
            for img_dir in img_dirs:
                self.img_dir_labels += [[os.path.join(img_dir, img_name), 1]
                                        for img_name in os.listdir(img_dir)]

        self.transform = transform
        self.target_transform = target_transform
        self.read_method = read_method
        self.images = []
        # directly load data into memory (for faster training)
        if in_memory:
            for info in self.img_dir_labels:
                img_path = info[0]
                image = self.read_transform_image(img_path)
                self.images.append(image)

    def read_transform_image(self, img_path):
        if self.read_method == 'pillow':
            image = Image.open(img_path)
            if image.mode == 'RGBA':
                image_new = Image.new('RGB', size=image.size, color=(255, 255, 255))
                image_new.paste(image, image)
                image = image_new
        elif self.read_method == 'torchvision.io':
            image = read_image(img_path)/255.0
        else:
            raise NotImplementedError
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_dir_labels)

    def __getitem__(self, idx):
        if not self.images:  # image not in memory
            image = self.read_transform_image(self.img_dir_labels[idx][0])
        else:  # already in memory
            image = self.images[idx]
        label = self.img_dir_labels[idx][1]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
