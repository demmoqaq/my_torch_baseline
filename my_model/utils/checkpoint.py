# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : checkpoint.py
# Author     ：Wang Yuhao
# Description：
"""
import torch
from my_model.utils.events import LOGGER


def load_state_dict(weights, model, map_location=None):
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model
