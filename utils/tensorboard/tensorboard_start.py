# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : tensorboard_start.py
# Author     ：Wang Yuhao
# Description：
"""
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

print('->all running records will be available in tensorboard')
writer = SummaryWriter(os.path.join(os.getcwd(),
                                    f"tensorboard_logs/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"))
print('    ->location of tensorboard log file is : ' +
      writer.log_dir + '\n----------------------------------------------------------------')
