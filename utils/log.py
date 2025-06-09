import os
import wandb
import time


class Wandb:
    def __init__(self,args,id):
        self.id = id
        self.args = args
        self.project = args.model_id
        self.name = f"{args.data}_{time.time()}"
        self.wandb_login()

    def wandb_login(self):
        os.system(f"wandb login --relogin {id}")
        wandb.init(
        project=self.project,
        name=self.name,
        config=vars(self.args),
        reinit=True
    )
    
    def log_metrics(self,metrics):
        wandb.log()
    
    