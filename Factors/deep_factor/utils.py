from dataset import MyDataset
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from collections import defaultdict
from loss import Loss
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Exp:
    def __init__(self, device, lr, single_num, epoches, print_mode=False, regular=False):

        self.net = None
        self.optimizer = None
        self.criterion = Loss()
        self.device = device
        self.print_mode = print_mode
        self.single_num = single_num
        self.epoches = epoches
        self.dic = defaultdict(list)
        self.train_data = []   #list of dataloader
        self.eval_data = []    #list of dataloader
        self.test_data = []    #list of dataloader 
        self.print_num = 1000
        self.regular = regular
        self.interval = 5

        self.data_len_each_day = []
        self.filter_list = []  #记录需要filter的数据集，主要方便验证集计算rank_ic
        self.dir_y_p = None
    

    def net_init(self):
        self.net.weight_init()
        self.net.to(self.device)


    def shuffle_date_lst(self, date_lst, back_days, frac=0.9):
        self.date_lst = date_lst
        # length = len(date_lst)
        # idx_shuffle = np.random.choice(length, length, replace=False)
        # self.date_lst = []
        # for idx in idx_shuffle:
        #     self.date_lst.append(date_lst[idx])
        date_len = len(date_lst) - back_days
        self.train_date = date_lst[ : int(date_len*frac)+back_days]
        self.eval_date = date_lst[int(date_len*frac)+back_days: ]


    def get_data_mode(self, dir_x, dir_y, back_days, batchsize, y_col_name, dir_f, mode='train'):
        pass


    def get_data(self, dir_x, dir_y, dir_f, back_interval, back_days, batchsize, y_col_name):
        print('#####-------------------get data-------------------#####')
        self.get_data_mode(dir_x, dir_y, back_interval, back_days, batchsize, y_col_name, dir_f, mode='train')
        self.get_data_mode(dir_x, dir_y, back_interval, back_days, batchsize, y_col_name, dir_f, mode='eval')


    def sparse(self, x):
        x = x.copy()
        x = x[np.arange(len(x)//self.interval)*self.interval]
        return x


    def data_scaler(self, x):
        '''批量读的话需要使用同一个scaler'''
        scaler = StandardScaler()
        scaler.fit(x)
        self.x_scaler = scaler


    def train(self, loss_type='mse', freeze=False,  trans_epoch=8):
        self.net.train()

        print('#####-------------------train-------------------#####')
        for epoch in range(self.epoches):
            if freeze and epoch == trans_epoch:
                self.net.freeze_layer()
                self.regular = False
                loss_type = 'ic'
                for params in self.optimizer.param_groups:
                    params['lr'] *= 1e-1
            print(f'------------epoch/epoches{epoch+1}/{self.epoches}------------')
            for i, dataloader in enumerate(self.train_data):
                print(f'train dataloader {i}:')
                for batch_idx, (x, y) in enumerate(dataloader):
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    y_pred = self.net(x)
                    loss = self.criterion(y, y_pred, type=loss_type, mean=True, regular=self.regular)
                    loss.backward()
                    self.optimizer.step()

                    if batch_idx % self.print_num == 0:
                        _, _, ic_mean, mse =  self._eval()
                        print(f'batch :{batch_idx}, train_{loss_type}: {loss: .6f}, eval_mse: {mse: .6f}, rank_ic: {ic_mean: .6f}')
                        self.net.train()


    def _eval(self):
        self.net.eval()
        rankic_es = []
        mse_es = 0
        cnt = 0
        for i, dataloader in enumerate(self.eval_data):
            y_pred_each_day, y_each_day = [], []
            data_len_each_day = self.data_len_each_day[i]
            filter_list_each_day = self.filter_list[i].all(axis=1)   #暂时只考虑当天出现涨停就去掉，带mask的rank_ic计算？
            cur_batch_idx = 0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y_pred = self.net(x).cpu().detach().numpy()
                y_pred_each_day.append(y_pred)
                y_each_day.append(y.numpy())
                if batch_idx+1 in data_len_each_day:
                    filter_tmp = filter_list_each_day[cur_batch_idx: batch_idx+1]
                    y_pred_each_day = torch.tensor(np.array(y_pred_each_day).reshape(-1, len(self.idx))[filter_tmp], dtype=torch.float)
                    y_each_day = torch.tensor(np.array(y_each_day).reshape(-1, len(self.idx))[filter_tmp], dtype=torch.float)  
                    
                    mse = self.criterion(y_pred_each_day, y_each_day, type='mse')
                    mse_es += mse 
                    cnt += 1
                    
                    rankic = self.criterion(y_pred_each_day, y_each_day, type='rank_ic', mean=False).detach().numpy()  #mean=False  ==> shape=[240//interval-2] 
                    rankic_es.append(rankic)

                    output_dir = os.path.join(self.dir_y_p, self.filter_ins)
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    # pd.DataFrame(y_pred_each_day.numpy()).to_csv(os.path.join(output_dir, '{}-{}.csv'.format(i, cur_batch_idx)))

                    y_pred_each_day, y_each_day = [], []
                    cur_batch_idx = batch_idx+1

        rankic_es = np.array(rankic_es)
        return rankic_es, rankic_es.mean()/rankic_es.std(), rankic_es.mean(), mse_es /cnt


    
    def eval(self):
        print('#####-------------------eval-------------------#####')
        rankic_es, icir, ic_mean, mse =  self._eval()
        print(f'eval set: mse: {mse: .6f}, eval_icir: {icir: .6f}, ic_mean: {ic_mean: .6f}')
        return rankic_es, icir, ic_mean
    

