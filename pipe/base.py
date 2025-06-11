import os
import torch
from models import TimeLLM

class Base(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeLLM':TimeLLM
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device = 'cuda'

            print(device)
        else:
            device = 'cpu'
            print('Use CPU')
        return device
          

      

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass