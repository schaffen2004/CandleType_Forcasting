from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import argparse
import torch
from utils.tools import del_files, EarlyStopping, adjust_learning_rate
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
import wandb
from datetime import datetime
from pipe.trainer import Trainer


if __name__== '__main__':
    # set random
    config.set_seed(2021)

    # get config
    args = config.get_args()
    args.session_id = f"{args.model}_{args.data}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Visualize arguments info
    display_args_table(args)
    
    # get wandb id
    load_dotenv("config/.env")  # Tải biến từ .env
    wandb_id = os.getenv("WANDB_API_ID")
    wandb = Wandb(args,wandb_id)
    
    # Training
    if args.is_training:
        trainer = Trainer(args)
        trainer.train()
        torch.cuda.empty_cache()
        
    else:
        pass
    
    
    
    



