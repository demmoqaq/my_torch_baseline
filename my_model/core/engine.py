# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : engine.py
# Author     ：Wang Yuhao
# Description：
"""
import time

import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

import tools.eval as eval
from utils.tensorboard.tensorboard_start import writer
from my_model.utils.events import load_yaml, LOGGER, NCOLS
from my_model.utils.ema import ModelEMA
from my_model.utils.checkpoint import load_state_dict
from my_model.data.data_entry import load_image_dataset
from my_model.model.model_entry import select_model
from my_model.model.loss import ComputeLoss
from my_model.solver.build import build_optimizer, build_lr_scheduler


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        if args.resume:
            self.ckpt = torch.load(args.resume, map_location='cpu')

        self.main_process = True
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

        self.model = model
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']

        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size

    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop()

        except:
            pass

    def train_in_loop(self):
        try:
            self.prepare_for_steps()
            for self.step, self.batch_data in self.pbar:
                self.train_in_steps()
        except Exception as _:
            LOGGER.error('ERROR in training steps.')


    def train_in_steps(self):
        images, targets = self.prepro_data(self.batch_data, self.device)
        # forward
        with amp.autocast(enabled=self.device != 'cpu'):
            preds = self.model(images)
            total_loss, loss_items = self.compute_loss(preds, targets)

        # backward
        self.scaler.scale(total_loss).backward()
        self.loss_items = loss_items
        self.update_optimizer()

    def eval_and_save(self):
        remaining_epochs = self.max_epoch - self.epoch
        eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 1
        is_val_epoch = (not self.args.eval_final_only or (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)
        if self.main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])
            if is_val_epoch:
                self.eval_model()
                self.

    def eval_model(self):
        results = eval.run(self.data_dict,
                           batch_size=self.batch_size,
                           img_size=self.img_size,
                           model=self.ema.ema,
                           dataloader=self.val_loader,
                           save_dir=self.save_dir,
                           task='train')
        LOGGER.info(f"Epoch: {self.epoch} | ")

    def train_before_loop(self):
        LOGGER.info("start training...")
        self.start_time = time.time()
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum), 1000)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')

        self.best_ap, self.ap = 0.0, 0.0
        self.evaluate_results = (0, 0)
        self.compute_loss = ComputeLoss()

        if hasattr(self, 'ckpt'):
            resume_state_dict = self.ckpt['model'].float().state_dict()
            self.model.load_state_dict(resume_state_dict, strict=True)
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.start_epoch = self.ckpt['epoch'] + 1
            self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
            self.ema.updates = self.ckpt['updates']

    def prepare_for_steps(self):
        if self.epoch > self.start_epoch:
            self.scheduler.step()
        self.model.train()
        self.loss = torch.zeros(1, device=self.device)
        self.optimizer.zero_grad()

        LOGGER.info(('\n' + '%10s' * 2) % ('Epoch', 'loss'))
        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    def print_details(self):
        pass

    def train_after_loop(self):
        pass

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                  [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step

    @staticmethod
    def get_data_loader(args, configs, data_dict, need_val):
        train_loader = load_image_dataset(data_dict['train_path'], data_dict['train_anno_path'], args, is_train=True)
        if need_val:
            val_loader = load_image_dataset(data_dict['val_path'], data_dict['val_anno_path'], args, is_train=False)
            return train_loader, val_loader
        else:
            return train_loader

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device, non_blocking=True).float()
        targets = batch_data[1].to(device)
        return images, targets

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

