import os
import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import sys

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# 再加上vwap=turnover.diff(n)/volume.diff(n)
# n=1，5，10，30，60，120 min

def df_complete(df, df_re_sz, df_re_sh, inst, date, is_sz, last_df=None, is_last=0):
    '''股票df完善:
        1、中间部分min没有成交记录的用前1min补充, 如果是刚开始就没成交记录的用前一天的最后一条补充; (由于原始数据已经补充了中间的, 这里只需补充开始和结尾没有成交记录的)
        2、复权因子处理 + 计算换手率
    '''

    #复权因子处理 + 计算换手率
    if is_sz:
        df['close'] = df['close'] * df_re_sz.loc[int(inst), date]  
    else:
        df['close'] = df['close'] * df_re_sh.loc[int(inst), date] 


    #flag=0
    '''case1: 开盘后一段时间没交易'''
    begin_time = str(int(df.iloc[0]['time']))[-4:]
    if begin_time != '0931':
        count1 = 0
        if begin_time > '1130':   #可能整个上午没交易
            if begin_time > '1301':  #可能下午刚开始也没交易
                count1 = int(begin_time[:2])*60 + int(begin_time[2:]) - (60*13+1)
            begin_time = '1131'
        count = int(begin_time[:2])*60 + int(begin_time[2:]) - (60*9+31) + count1
        # print('begin:', count)
        # if count > 120:
        #     print(df.head())
        #     flag = 1
        if is_last:
            df1 = last_df.iloc[-1:, :].reset_index(drop=True)
        else:
            df1 = df.iloc[0:1, :].reset_index(drop=True)
        df1.loc[0, ['open', 'high', 'low']] = df1.loc[0, 'close']
        df1.loc[0, ['volume', 'turnover']] = 0
        df = pd.concat([df1]*count + [df])
    
    '''case2: 收盘前一段时间没交易'''
    end_time = str(int(df.iloc[-1]['time']))[-4:]
    if end_time != '1500':
        count1 = 0
        if end_time < '1301':   #可能整个下午没有交易
            if end_time < '1130':  #可能早上后半段都没交易
                count1 = 11*60 + 30 - (int(end_time[:2])*60 + int(end_time[2:]) )
            end_time = '1300'
        count = 15*60 - (int(end_time[:2])*60 + int(end_time[2:]) ) + count1
        # print('end:', count)
        # if count > 120:
        #     print(df.tail())
        #     flag = 1
        df1 = df.iloc[-1:, :].reset_index(drop=True)
        df1.loc[0, ['open', 'high', 'low']] = df1.loc[0, 'close']
        df1.loc[0, ['volume', 'turnover']] = 0
        df = pd.concat([df] + [df1]*count)   
    assert len(df) == 240


    if is_last:
        #delete last three rows in last_df
        # df_last = last_df.iloc[:-3].copy()
        #complete
        # df1 = df_last.iloc[-1::].reset_index(drop=True)
        # df1.loc[0, ['open', 'high', 'low']] = df1['close'].iloc[0]
        # df1.loc[0, ['volume', 'turnover']] = 0
        # df_last = pd.concat([df_last] + [df1]*3)
        # assert len(df_last) == 240
        df = pd.concat([last_df, df])
        
    # df.reset_index(drop=True, inplace=True)
    return df[-240:]



begin_date = sys.argv[1]
end_date= sys.argv[2]
dir_k_line = '/net_246_public_data/1minbar'
output_dir = '/home/largefile/public/zjwang/HF_factors/kline_feature/output'
date_lst = os.listdir(dir_k_line)
date_lst.sort()
date_lst = [date for date in date_lst if date >= begin_date and date <= end_date]
print(date_lst)


#复权因子
df_re_sz = pd.read_csv('/home/largefile/public/zjwang/eoddata/SZAdjF.csv', index_col=0)
df_re_sh = pd.read_csv('/home/largefile/public/zjwang/eoddata/SHAdjF.csv', index_col=0)
df_re_sz.fillna(1, inplace=True)
df_re_sh.fillna(1, inplace=True)


dic = defaultdict(pd.DataFrame)
intervals = [1, 5, 10, 30, 60, 120]
for date in tqdm(date_lst):
    dir = os.path.join(dir_k_line, date)
    for stock in os.listdir(dir):
        try:
            if stock[7:9] == 'SH':
                is_sz = 0
                if stock[:2] not in ['60', '68']:
                    continue
            if stock[7:9] == 'SZ':
                is_sz = 1
                if stock[:2] not in ['00', '30']:
                    continue
            df = pd.read_csv(os.path.join(dir, stock)) 
            if stock not in dic:
                df = df_complete(df, df_re_sz, df_re_sh, stock[:6], date, is_sz)
                dic[stock] = df
                continue
            df_last = dic[stock]
            df = df_complete(df, df_re_sz, df_re_sh, stock[:6], date, is_sz, df_last, is_last=1)
        
            df_concat = pd.concat([df_last, df])
            dic[stock] = df
            #vwap
            for interval in intervals:
                series1 = df_concat['volume'].rolling(interval).sum()
                series2 = df_concat['turnover'].rolling(interval).sum()
                df_concat[f'vwap_{interval}'] = series2 / series1
            df = df_concat.iloc[-240:, :]

            #昨收
            pre_close = df_last['close'].iloc[-1]
            df['pre_close'] = pre_close
            df.loc[:, 'pre_close_log'] = np.log(df['pre_close'])

            #涨幅
            df['percentage_change'] = df['close'] / pre_close
            df['close_log'] = np.log(df['close'])

            df = df.iloc[:, 3:]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(os.path.join(output_dir, date)):
                os.mkdir(os.path.join(output_dir, date))
            df.to_csv(os.path.join(output_dir, date, stock), index=False)
        except:
            print(f'{date}-{stock} failed')

