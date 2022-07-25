# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : events.py
# Author     ：Wang Yuhao
# Description：
"""
import yaml


def load_yaml(file_path):
    """load_yaml(file_path) -> yaml"""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict
