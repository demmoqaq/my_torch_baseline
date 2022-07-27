# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : events.py
# Author     ：Wang Yuhao
# Description：
"""
import os
import yaml
import logging
import shutil


def set_logger(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logger(__name__)
NCOLS = shutil.get_terminal_size().columns


def load_yaml(file_path):
    """load_yaml(file_path) -> yaml"""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict
