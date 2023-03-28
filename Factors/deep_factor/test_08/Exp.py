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
from model import mlp, lstm
import pickle


class Exp:
    def __init__(self, params, params_net):

        #define model
        self.model_name = params['model']
        self.params_net = params_net
            
        #define criterion
        self.lr = params['lr']

        #define loss
        self.criterion = Loss()
        self.criterion.alpha = params['alpha']

        self.device = torch.device(f"cuda:{params['device']}" if torch.cuda.is_available() else 'cpu')  
        self.input_dim = 0

        #define some dirs
        self.dir_x = params['dir_x']    #input x
        self.dir_y = params['dir_y']    #input y
        self.dir_f = params['dir_f']    # filter rule dir
        self.dir_y_p = params['dir_y_p']  #dir to output yp


        #if filter data
        self.filter_mode = False
        if 'is_filter' in params and params['is_filter']:
            self.get_filter_csv(self.dir_f)
            self.filter_mode = True
            self.filter_ins = params['ins']
        
        #if trans your loss
        self.epoches = params['epoches']
        self.freeze = params['freeze']     # freeze layers except last linear layer
        self.lr_decay = params['lr_decay']  # if lr decay after change loss
        self.trans_epoch = params['trans_epoch']  # epoch to trans loss


        #some setting
        self.single_num = params['single_num']  #max days of single dataloader
        self.print_mode = params['print_mode']


        self.print_num = params['print_num']  
        self.regular = params['regular']    #if add regular loss (l1 loss)
        self.y_col_name = params['y_col_name']
        self.batch_size = params['batch_size']
        
        self.date_len = params['date_len']   # the date len of data
        self.interval = params['interval']    #data sample rate 
        self.exp_idx = params['exp_idx']
        if self.interval:
            self.idx = np.arange(5, 211, self.interval)   #idx to use each day
        else:
            self.idx = np.array([params['idx']])


        self.back_days = params['back_days']      #day
        self.back_interval = params['back_interval']  #back_interval feature (min)
        self.interval_feature = params['interval_feature']  #feature sample rate  
        self.train_data_frac = params['train_data_frac']
        self.shuffle_date_lst()

        #some dataset
        self.dic = defaultdict(list)
        self.train_data = []   #list of dataloader
        self.eval_data = []    #list of dataloader
        self.test_data = []    #list of dataloader 
        self.data_len_each_day = []
        self.stocks_list = []
        self.filter_list = []  #record filter mask to calculate rank_ic each day

        self.output_dir = os.path.join(self.dir_y_p, self.model_name + '_' + self.filter_ins + '_' + self.exp_idx)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        
    def net_init(self):
        if self.model_name == 'mlp':
            self.params_net['input_dim'] = self.back_interval // self.interval_feature * self.input_dim
            self.net = mlp(self.params_net)
        if self.model_name == 'lstm':
            self.params_net['input_dim'] = self.input_dim
            self.net = lstm(self.params_net)
        self.net.weight_init()
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion.model = self.net


    def shuffle_date_lst(self):
        #todo: shuffle globally 
        self.date_lst = os.listdir(self.dir_x)
        self.date_lst.sort()
        self.test_date = self.date_lst[-10:]
        self.date_lst = self.date_lst[-(self.date_len+self.back_days)-10: -10]

        self.train_date = self.date_lst[ : int(self.date_len*self.train_data_frac)+self.back_days]
        self.eval_date = self.date_lst[int(self.date_len*self.train_data_frac)+self.back_days: ]


    def get_filter_csv(self, dir):
        df_st = pd.read_csv(os.path.join(dir, 'ashareIsST_1801_2209.csv'), index_col=0)
        self.filter_df_st = df_st

        df_list = pd.read_csv(os.path.join(dir, 'list_less20.csv'), index_col=0)
        self.filter_df_list = df_list

        df_amt = pd.read_csv(os.path.join(dir, 'amt_less_2.csv'), index_col=0)
        self.filter_df_amt = df_amt


    def get_data(self):
        print('#####-------------------get data-------------------#####')
        self.get_data_mode(mode='train')
        self.net_init()
        self.get_data_mode(mode='eval')


    def train(self, loss_type='mse'):
        self.net.train()

        print('#####-------------------train-------------------#####')
        for epoch in range(self.epoches):
            if self.freeze and epoch == self.trans_epoch-1:
                self.net.freeze_layer()
                loss_type = 'ic'
                if self.lr_decay:
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


    def _eval(self, final=False):
        self.net.eval()
        rankic_es = []
        mse_es = 0
        cnt = 0
        for i, dataloader in enumerate(self.eval_data):
            y_pred_each_day, y_each_day = [], []
            data_len_each_day = self.data_len_each_day[i]
            stocks_list = self.stocks_list[i]
            filter_list_each_day = self.filter_list[i].all(axis=1)   #暂时只考虑当天出现涨停就去掉，todo: 带mask的rank_ic计算？
            cur_batch_idx = 0
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y_pred = self.net(x).cpu().detach().numpy()
                y_pred_each_day.append(y_pred)
                y_each_day.append(y.numpy())
                if batch_idx+1 in data_len_each_day:
                    filter_tmp = filter_list_each_day[cur_batch_idx: batch_idx+1]
                    stocks = stocks_list[cur_batch_idx: batch_idx+1]
                    y_pred_each_day = torch.tensor(np.array(y_pred_each_day).reshape(-1, len(self.idx))[filter_tmp], dtype=torch.float)
                    y_each_day = torch.tensor(np.array(y_each_day).reshape(-1, len(self.idx))[filter_tmp], dtype=torch.float)  
                    
                    mse = self.criterion(y_pred_each_day, y_each_day, type='mse')
                    mse_es += mse 
                    
                    rankic = self.criterion(y_pred_each_day, y_each_day, type='rank_ic', mean=False).detach().numpy()  #mean=False  ==> shape=[240//interval-2] 
                    rankic_es.append(rankic)

                    if final:
                        stocks = np.array(stocks)[filter_tmp]
                        pd.DataFrame(y_pred_each_day.numpy(), index=stocks).to_csv(os.path.join(self.output_dir, '{}.csv'.format(self.eval_date[cnt])))
                        

                    y_pred_each_day, y_each_day = [], []
                    cur_batch_idx = batch_idx+1
                    cnt += 1

        rankic_es = np.array(rankic_es)
        return rankic_es, rankic_es.mean()/rankic_es.std(), rankic_es.mean(), mse_es/cnt

    
    def eval(self):
        print('#####-------------------eval-------------------#####')
        rankic_es, icir, ic_mean, mse =  self._eval(final=True)
        print(f'eval set: mse: {mse: .6f}, eval_icir: {icir: .6f}, ic_mean: {ic_mean: .6f}')
        return rankic_es, icir, ic_mean
    

#%%  data process


    def get_data_mode(self,  mode='train'):       
        cur_cnt = 0
        if mode == 'train':
            date_lst_all = self.train_date
        elif mode == 'eval':
            date_lst_all = self.eval_date
        elif mode == 'test':
            date_lst_all = self.test_date

        print(f'get {mode} data of {len(date_lst_all)} days(containing {self.back_days} days of backdays if mode is train)')

        while cur_cnt*self.single_num < len(date_lst_all):
            date_lst = date_lst_all[cur_cnt*self.single_num: (cur_cnt+1)*self.single_num]
            
            input_x, input_y = [], []
            data_len_each_day, filter_list_each_dataloader, stocks_list = [], [], []
            for i in tqdm(range(len(date_lst))):
                filter_list_one_day = []
                date = date_lst_all[cur_cnt*self.single_num+i]
                if self.print_mode: print(f'-{date}-')
                if cur_cnt*self.single_num+i == 0:
                    df_limit = pd.read_csv(os.path.join(self.dir_f, 'limit_minbar_daily', 'limit_{}.csv'.format(date)), index_col=0)
                else:
                    df_limit = future_df_limit
                if cur_cnt*self.single_num+i+1 < len(date_lst_all):
                    furture_date = date_lst_all[cur_cnt*self.single_num+i+1]
                    future_df_limit = pd.read_csv(os.path.join(self.dir_f, 'limit_minbar_daily', 'limit_{}.csv'.format(furture_date)), index_col=0)
                date_dir = os.path.join(self.dir_y, date) #用y的，因为在生成时候future没有会跳过，y的少一点。
                for file in os.listdir(date_dir):
                    if file.split('.')[1] != self.filter_ins:
                        continue
                    data = pd.read_csv(os.path.join(self.dir_x, date, file)).fillna(0)
                    #=================================================
                    #去掉一些因子
                    columns = data.columns
                    conditon = (columns.str.startswith('var') | columns.str.startswith('up_fluctuate') | columns.str.startswith('down_fluctuate') ) 
                    data = data.drop(columns=columns[conditon])
                    #=================================================
                    data = data.values
                    data[np.isinf(data)] = 0
                    self.dic[file].append(np.concatenate([data, np.arange(240).reshape(240, -1)], axis=1))     #加上时间特征
                    if not self.input_dim:
                        self.input_dim = data.shape[-1] + 1
                    if len(self.dic[file]) > self.back_days:
                        if mode == 'eval':
                            stocks_list.append(file[:-4])
                        filter_list_one_day.append(self.filter(date, file, df_limit, future_df_limit, mode))
                        y_data = pd.read_csv(os.path.join(self.dir_y, date, file))[-240:].fillna(0)
                        tmp_y = y_data[self.y_col_name].values
                        tmp_y[np.isinf(tmp_y)] = 0
                        tmp_x = np.concatenate(self.dic[file][:self.back_days+1], axis=0) # shape:[240*(back_days+1), feature_len=factors_len + 1]
                        self.dic[file] = self.dic[file][1:]
                        self.input_data(input_x, input_y, tmp_x, tmp_y, self.back_interval, file, date)


                if len(input_x):   
                    data_len_each_day.append(len(input_x))
                    filter_list_each_dataloader.append(np.array(filter_list_one_day))
            
            print('data_len_each_day:', data_len_each_day)
                

            self.feature_len = input_x[0].shape[-1]
            filter_list_each_dataloader = np.concatenate(filter_list_each_dataloader, axis=0)
            input_x = np.array(input_x).reshape(-1, self.feature_len)
            input_y = np.array(input_y).reshape(-1, 1)
            if mode == 'train':  #如果是train，直接过滤掉这些股票
                filter_list_each_dataloader = filter_list_each_dataloader.reshape(-1)
                input_x = input_x[filter_list_each_dataloader]
                input_y = input_y[filter_list_each_dataloader]
                if cur_cnt == 0:
                    #scaler只能对2D数据
                    self.data_scaler(input_x)
            input_x = self.x_scaler.transform(input_x)
            if self.model_name == 'lstm':
                input_x = input_x.reshape(-1, self.back_interval//self.interval_feature, self.feature_len*self.interval_feature//self.back_interval)

            print('Estimated number of stocks', len(input_x)//len(self.idx))
            input_x, input_y = torch.tensor(input_x, dtype=torch.float), torch.tensor(input_y, dtype=torch.float)
            dataset = MyDataset(input_x, input_y)



            if self.print_mode:
                print(f'get {mode} data with length of {len(input_x)}')


            if mode == 'train':
                torch.manual_seed(0)
                self.train_data.append(DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=4, drop_last=True))
                # self.train_data = DataLoader(dataset=list(zip(input_x, input_y)), batch_size=batchsize, shuffle=True, pin_memory=False)
            elif mode == 'eval':
                self.eval_data.append(DataLoader(dataset=dataset, batch_size=len(self.idx), shuffle=False, pin_memory=False, num_workers=4, drop_last=True))
                self.data_len_each_day.append(data_len_each_day)
                self.filter_list.append(filter_list_each_dataloader)
                self.stocks_list.append(stocks_list)
                # self.eval_data = DataLoader(dataset=list(zip(input_x, input_y)), batch_size=240, shuffle=True, pin_memory=False)
            elif mode == 'test':
                self.test_data.append(DataLoader(dataset=dataset, batch_size=len(self.idx), shuffle=False, pin_memory=False, num_workers=4, drop_last=True))
                self.data_len_each_day.append(data_len_each_day)
                self.filter_list.append(filter_list_each_dataloader)
                self.stocks_list.append(stocks_list)
                # self.test_data = DataLoader(dataset=list(zip(input_x, input_y)), batch_size=240, shuffle=True, pin_memory=False)
            
            cur_cnt += 1

        if mode != 'train':
            self.dic = defaultdict(list)


    def input_data(self, input_x, input_y, x, y, back_interval, file, date):
        #x shape: [240*back_days, feature_len]
        #y shape: [240, 1]
        rolling_x = self.rolling(x, back_interval, file, date)
        input_x.append(rolling_x[self.idx])
        input_y.append(y[self.idx])
        


    def rolling(self, a, window, file, date, concat=True):
        #对array按window对第一个维度进行rolling，concat=True对过去的window进行concat，否则得到一个序列
        _a = a.copy()
        shape_a = _a.shape
        shape = [shape_a[0]-window+self.interval_feature, window//self.interval_feature, shape_a[-1]]
        strides = [_a.strides[0], _a.strides[0]*self.interval_feature, _a.strides[1]]
        a_rolling = np.lib.stride_tricks.as_strided(_a, shape=shape, strides=strides)
        if concat:
            a_rolling = a_rolling.reshape(shape[0], -1)
        return a_rolling[-240:]


    def data_scaler(self, x):
        '''批量读的话需要使用同一个scaler'''
        scaler = StandardScaler()
        scaler.fit(x)
        self.x_scaler = scaler
        pickle.dump(self.x_scaler, open(os.path.join(self.output_dir, 'scaler.pkl'), 'wb'))


    def filter(self, date, file, df_limit, future_df_limit, mode='train'):
        #file 是 csv结尾
        #df_limit 是 日度的
        #future_limit 是第二天的涨跌停，持仓到期的时候涨跌停也去掉
        #输出是否保留 的 array
        stock_name = file[:-4]
        try:
            if self.filter_df_st.loc[int(date), stock_name]:
                return np.array([False]*(len(self.idx)))
            if self.filter_df_amt.loc[int(date), stock_name]:
                return np.array([False]*(len(self.idx)))
            if self.filter_df_list.loc[int(date), stock_name]:
                return np.array([False]*(len(self.idx)))
            
            #涨跌停，训练集去掉当前和未来30min涨跌的，其他的只去掉当前的，不跨日不需要未来的
            df_limit_tmp = df_limit[stock_name] 
            # future_limit = future_df_limit[stock_name][indx]
            cur_limit = df_limit_tmp[self.idx]
            cur_limit_shift = df_limit_tmp.shift(-30).fillna(0)[self.idx]
            if mode == 'train':
                condition = ~np.logical_or(cur_limit, cur_limit_shift).values
                return condition
            elif mode == 'eval' or mode == 'test':
                condition = (cur_limit == 0).values
                return condition
            else:
                return np.array([True]*(len(self.idx)))
        except: #有可能不在这个df里面 保留
            return np.array([True]*(len(self.idx)))
        

    def exp(self):
        self.get_data()
        self.train()
        res, icir, ic_mean = self.eval()
        return res, icir, ic_mean