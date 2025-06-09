from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
import torch
# from models import Autoformer, DLinear, TimeLLM
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from dataset import loader,provider
import random
import numpy as np
import os
import time
from dataset import provider
from config import config   
from dataset import provider
from dotenv import load_dotenv
from utils.log import Wandb
from utils.visualization import display_args_table
if __name__== '__main__':
    # set random
    config.set_seed(2021)

    # get config
    args = config.get_args()
    
    # get wandb id
    load_dotenv("config/.env")  # Tải biến từ .env
    wandb_id = os.getenv("WANDB_ID")
    
    # Visualize arguments info
    display_args_table(args)
    
    wandb = Wandb(args,wandb_id)
    
    
    



