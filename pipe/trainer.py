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
from utils.tools import EarlyStopping,adjust_learning_rate
from torchmetrics import Precision, Recall, F1Score, Accuracy
from utils.visualization import print_eval
warnings.filterwarnings('ignore')


class Trainer(Base):
    def __init__(self, args):
        super(Trainer, self).__init__(args) 

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
             
        model = nn.DataParallel(model, device_ids=[0,1])
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    

    def _select_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    
    def train(self):
        train_data, train_loader = provider(self.args, 'train')
        vali_data, vali_loader = provider(self.args, 'val')
        test_data, test_loader = provider(self.args, 'test')
        

        path = os.path.join(self.args.checkpoints,self.args.session_id)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            # Khởi tạo metrics trên GPU
            precision_metric = Precision(task="binary", threshold=0.5).to(self.device)
            recall_metric = Recall(task="binary", threshold=0.5).to(self.device)
            f1_metric = F1Score(task="binary", threshold=0.5).to(self.device)
            accuracy_metric = Accuracy(task="binary", threshold=0.5).to(self.device)
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
                wandb.log({"Epoch": epoch + 1})
                
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
              
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                dec_inp = None
             
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
     
                outputs = outputs[:, -self.args.pred_len:, -1]
                loss = criterion(outputs, batch_y)

                
                # Cập nhật metrics
                precision_metric.update(outputs, batch_y.int())
                recall_metric.update(outputs, batch_y.int())
                f1_metric.update(outputs, batch_y.int())
                accuracy_metric.update(outputs, batch_y.int())
                
                train_loss.append(loss.item())
                
                wandb.log({"train_loss/iteration": loss.item()})
                
                
                loss.backward()
                model_optim.step()
                
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()
                       

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            precision = precision_metric.compute().item()
            recall = recall_metric.compute().item()
            f1 = f1_metric.compute().item()
            accuracy = accuracy_metric.compute().item()


            # In kết quả
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Average Loss: {train_loss:.4f}")
            
            val = self.val(vali_data, vali_loader, criterion)
            test = self.val(test_data, test_loader, criterion)
            

            
            wandb.log({'train_loss': train_loss, 'vali_loss': val['loss'], "test_loss": test['loss']})
            wandb.log({'train_acc': accuracy, 'vali_acc': val['acc'], "test_acc": test['acc']})
            # wandb.log({'train_precision': precision, 'vali_precision': val['pre'], "test_precision": test['pre']})
            # wandb.log({'train_recall': recall, 'vali_recall': val['recall'], "test_recall": test['recall']})
            # wandb.log({'train_F1': f1, 'vali_F1': val['f1'], "test_F1": test['f1']})
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, val['loss'],test['loss']))
            
            early_stopping(val['loss'], self.model, path)
            
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        
    def val(self, val_data, val_loader, criterion):
        total_loss = []
        # Khởi tạo metrics trên GPU
        precision_metric = Precision(task="binary", threshold=0.5).to(self.device)
        recall_metric = Recall(task="binary", threshold=0.5).to(self.device)
        f1_metric = F1Score(task="binary", threshold=0.5).to(self.device)
        accuracy_metric = Accuracy(task="binary", threshold=0.5).to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(val_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                dec_inp = None

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs[:, -self.args.pred_len:, -1]
                

                precision_metric.update(outputs, batch_y.int())  # true cần là int cho torchmetrics
                recall_metric.update(outputs, batch_y.int())
                f1_metric.update(outputs, batch_y.int())
                accuracy_metric.update(outputs, batch_y.int())


                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1 = f1_metric.compute().item()
        accuracy = accuracy_metric.compute().item()


        # In kết quả
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Average Loss: {total_loss:.4f}")
        self.model.train()
        return {
            'loss':total_loss,
            'acc':accuracy,
            'pre':precision,
            'recall':recall,
            'f1':f1
        }
    
    def test(self,test_data, test_loader, criterion):
        total_loss = []        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = None

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1

                pred = outputs.detach()
                true = batch_y.detach()


                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss