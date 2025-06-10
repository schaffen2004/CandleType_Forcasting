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
        # Kiểm tra xem có GPU có sẵn không
        if torch.cuda.is_available():
            # Sử dụng GPU đầu tiên nếu không có yêu cầu cụ thể
            device = torch.device('cuda:0')  # Chọn GPU đầu tiên
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Chỉ định GPU 0
        else:
            # Nếu không có GPU, sử dụng CPU
            device = torch.device('cpu')
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Không sử dụng GPU
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass