# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model_entry.py
# Author     ：Wang Yuhao
# Description：
"""
from my_model.model.model_name import MyModel


def select_model(args, cfg, device):
    my_models = {
        'model1': MyModel(cfg).to(device),
    }
    _model_used = my_models[args.model_type]
    return _model_used
