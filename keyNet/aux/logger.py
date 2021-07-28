r"""Logging"""
import datetime
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.network_version +  logtime
        cls.logpath = os.path.join(args.weights_dir  +'/' +  logpath  + '.log')

        os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # # Tensorboard writer
        # cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # # Log arguments
        logging.info('\n+=========== Key.Net PyTorch Version ============+')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
        logging.info('+================================================+\n')

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)

    @classmethod
    def save_model(cls, model, epoch, val_rep):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. Repeability score: %5.2f.\n' % (epoch, val_rep))

