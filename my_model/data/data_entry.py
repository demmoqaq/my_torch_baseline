# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data_entry.py
# Author     ：Wang Yuhao
# Description：
"""
from torch.utils.data import DataLoader

from my_model.data.custom_dataset import CustomImageDataset
from my_model.data.data_transform import transform


def load_image_dataset(args, is_train):
    without_label = False
    if args.annotations_file_dirs is None:
        print("do not pass annotations_file_dir, will read data directly from directory, default label is 1")
        without_label = True
    dataset = CustomImageDataset(
        img_dirs=args.img_dirs, annotations_file_dirs=args.annotations_file_dirs,
        without_label=without_label, in_memory=args.data_in_memory, read_method=args.data_read_method,
        transform=transform, target_transform=None
    )

    if is_train:
        shuffle = True
        batch_size = args.batch_size
    else:
        shuffle = False
        batch_size = 1

    loader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=args.cpu_num, pin_memory=True, drop_last=False)
    return loader
