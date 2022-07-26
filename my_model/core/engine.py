# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : engine.py
# Author     ：Wang Yuhao
# Description：
"""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.tensorboard.tensorboard_start import writer
from my_model.utils.events import load_yaml, LOGGER
from my_model.utils.ema import ModelEMA
from my_model.utils.checkpoint import load_state_dict
from my_model.data.data_entry import load_image_dataset
from my_model.model.model_entry import select_model
from my_model.solver.build import build_optimizer, build_lr_scheduler


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        if args.resume:
            self.ckpt = torch.load(args.resume, map_location='cpu')

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir

        # get_data_loader
        self.data_dict = load_yaml(args.data_path)
        if self.args.need_val:
            self.train_loader, self.val_loader = self.get_data_loader(args, cfg, self.data_dict, self.args.need_val)
        else:
            self.train_loader = self.get_data_loader(args, cfg, self.data_dict, self.args.need_val)

        # get model and optimizer
        model = self.get_model(args, cfg, device)
        if args.tensorboard:
            _data_sample, _ = next(iter(self.train_loader))
            writer.add_graph(model, _data_sample)
            writer.close()
        self.optimizer = self.get_optimizer(args, cfg, model)
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
        self.ema = ModelEMA(model) if self.main_process else None

        self.start_epoch = 0
        # resume
        if hasattr(self, 'ckpt'):
            resume_start_dict = self.ckpt['model'].float().state_dict()
            model.load_state_dict(resume_start_dict, strict=True)
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']

        self.model = self.parallel_model(args, model, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']

        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size

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
        model = select_model(args, cfg, device)
        weights = cfg.model.pretrained
        if weights:  # finetune if pretrained model is set
            LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
            model = load_state_dict(weights, model, map_location=device)
        LOGGER.info('Model: {}'.format(model))
        return model

    @staticmethod
    def get_optimizer(args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        optimizer = build_optimizer(cfg, model)
        return optimizer

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf

    @staticmethod
    def parallel_model(args, model, device):
        # If DP mode
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)

        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

        return model
