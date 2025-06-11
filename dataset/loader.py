import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class XAUUSD(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='ETTh1.csv',
                 target='CandleType', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        
        if size == None:
            self.seq_len = 1024
            self.label_len = 0
            self.pred_len = 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        df_get = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # Lấy 1/4 dữ liệu cuối
        df_raw  = df_get.iloc[-len(df_get)//4:]
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('Time')
        
        df_timestamp = pd.DataFrame(df_raw['Time'])
        df_feattures = df_raw[cols]
        df_target = np.array(df_raw[self.target])

        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        

      
        train_features = df_feattures[border1s[0]:border2s[0]]
        
        self.scaler.fit(train_features.values)
        features_trans = self.scaler.transform(df_feattures.values)


        df_stamp = df_timestamp[border1:border2]
        df_stamp['Time'] = pd.to_datetime(df_stamp.Time)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.Time.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.Time.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.Time.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.Time.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Time'], 1).values
            
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


        self.data_x = np.concatenate((features_trans[border1:border2],df_target.reshape(-1,1)[border1:border2]),axis=1)
        self.data_y = df_target[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)