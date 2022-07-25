# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : engine.py
# Author     ：Wang Yuhao
# Description：
"""
import time

from utils.events import load_yaml
from my_model.data.data_entry import load_image_dataset


class Trainer:
    def __init__(self, args, configs, device):
        self.args = args
        self.configs = configs
        self.device = device

        # get_data_loader
        self.data_dict = load_yaml(args.data_path)
        if self.args.need_val:
            self.train_loader, self.val_loader = self.get_data_loader(args, configs, self.data_dict, self.args.need_val)
        else:
            self.train_loader = self.get_data_loader(args, configs, self.data_dict, self.args.need_val)

        # get model and optimizer
        model = self.get_model(args, configs, device)

    @staticmethod
    def get_data_loader(args, configs, data_dict, need_val):
        train_loader = load_image_dataset(data_dict['train_path'], data_dict['train_anno_path'], args, is_train=True)
        if need_val:
            val_loader = load_image_dataset(data_dict['val_path'], data_dict['val_anno_path'], args, is_train=False)
            return train_loader, val_loader
        else:
            return train_loader

    @staticmethod
    def get_model(args, cfg, device):
        model = build_model(config=cfg, device=device)
        return model
