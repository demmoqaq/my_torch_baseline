# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data_entry.py
# Author     ：Wang Yuhao
# Description：
"""
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from my_model.data.custom_dataset import CustomImageDataset
from my_model.data.data_transform import train_transform, no_transform
from utils.tensorboard.tensorboard_start import writer


def load_image_dataset(img_dirs, annotations_file_dirs, args, is_train):
    print('\r->creating data loader...')
    without_label = False
    if args.annotations_file_dirs is None:
        print("    ->do not pass annotations_file_dir, will read data directly from directory, default label is 1")
        without_label = True
    dataset = CustomImageDataset(
        img_dirs=img_dirs, annotations_file_dirs=annotations_file_dirs,
        without_label=without_label, in_memory=args.data_in_memory, read_method=args.data_read_method,
        transform=train_transform, target_transform=None
    )

    if is_train:
        shuffle = True
        batch_size = args.batch_size
        tensorboard_info = 'example of one batch data in training'
    else:
        shuffle = False
        batch_size = 1
        tensorboard_info = 'example of one batch data (not used in training)'

    loader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=args.cpu_num, pin_memory=True,
                        drop_last=False)
    if args.tensorboard:
        dataset_origin = CustomImageDataset(
            img_dirs=img_dirs, annotations_file_dirs=annotations_file_dirs,
            without_label=without_label, in_memory=False, read_method='pillow',
            transform=no_transform, target_transform=None
        )
        loader_origin = DataLoader(dataset_origin, batch_size, shuffle=shuffle, num_workers=args.cpu_num,
                                   pin_memory=True, drop_last=False)
        images_origin, _ = next(iter(loader_origin))
        img_grid_origin = make_grid(images_origin)
        writer.add_image('one batch of origin image (before transform)', img_grid_origin)

        images, _ = next(iter(loader))
        img_grid = make_grid(images)
        writer.add_image(tensorboard_info, img_grid)
        writer.close()
        print('    ->example of one batch data has been added to tensorboard')

    print('dataloader has been created!\n----------------------------------------------------------------')
    return loader
