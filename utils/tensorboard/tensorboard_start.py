# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : tensorboard_start.py
# Author     ：Wang Yuhao
# Description：
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from my_model.data.custom_dataset import CustomImageDataset
from collections import namedtuple
from torchvision import transforms
from torchvision.utils import make_grid
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# setup start fold
writer = SummaryWriter('./runs/xxx_experiment_1')

# setup dataloader which we want to visualize
args = namedtuple('args', 'img_dir annotations_file_dir in_memory') \
    ('../../test/data/val', '../../test/data/label.csv', False)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = CustomImageDataset(args, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

dataiter = iter(dataloader)
imgs, labels = dataiter.next()

img_grid = make_grid(imgs)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

matplotlib_imshow(img_grid, one_channel=True)

writer.add_image('four_fashion_mnist_images', img_grid)