from torch.optim import lr_scheduler
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from pipe.base import Base
from dataset.provider import provider

warnings.filterwarnings('ignore')


class Trainer(Base):
    def __init__(self, args):
        super(Trainer, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def train(self):
        pass

    def val(self):
        pass
    
    def test(self):
        pass