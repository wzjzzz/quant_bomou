import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from functools import reduce
import sys

# %%
# 补全原始df
def df_complete(df, df_re_sz, df_re_sh, df_free, inst, date, is_sz, last_df=None, is_last=0):


    # 复权因子处理 + 计算换手率
    if is_sz:
        df['close'] = df['close'] * df_re_sz.loc[int(inst), date] 
        df['turnover_rate'] = df['volume'] / df_free.loc[inst+'.SZ', 'FLOAT_A_SHR']   
    else:
        df['close'] = df['close'] * df_re_sh.loc[int(inst), date]
        df['turnover_rate'] = df['volume'] / df_free.loc[inst+'.SH', 'FLOAT_A_SHR']   


    #flag=0
    begin_time = str(int(df.iloc[0]['time']))[-4:]
    if begin_time != '0931':
        count1 = 0
        if begin_time > '1130': 
            if begin_time > '1301':
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
    

    end_time = str(int(df.iloc[-1]['time']))[-4:]
    if end_time != '1500':
        count1 = 0
        if end_time < '1301': 
            if end_time < '1130': 
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



# %% 
date_lst = os.listdir('/home/largefile/public/zjwang/HF_factors/f12/output_factors/lastday')
hold_time = int(sys.argv[1])
date_begin = sys.argv[2]
date_end = sys.argv[3]

# 补齐
# date_lst = date_lst[-45: ]
# print(date_lst)
date_lst.sort()
date_lst = [date for date in date_lst if date_begin <= date <= date_end]


#复权因子
df_re_sz = pd.read_csv('/home/largefile/public/zjwang/eoddata/SZAdjF.csv', index_col=0)
df_re_sh = pd.read_csv('/home/largefile/public/zjwang/eoddata/SHAdjF.csv', index_col=0)
df_re_sz.fillna(1, inplace=True)
df_re_sh.fillna(1, inplace=True)


last_df_dic = defaultdict(pd.DataFrame) 
future_df_dic = defaultdict(pd.DataFrame)


ignore_stock_lst = ['689009.SH.csv'] #有些股票第二个交易日没有成交记录，忽略这样的股票。默认这个股票忽略，因为生成因子的时候没有复权因子，所以还没生成。

for i in tqdm(range(len(date_lst))):

    date = date_lst[i]

    year, month = date[:4], date[4:6]
    #流通股本
    df_free = pd.read_csv(os.path.join('/home/largefile/public/zjwang/AShareEOD', year, month, 'ASHAREEOD.{}.csv'.format(date)))
    df_free['code'] = df_free['ticker'].map(lambda x:str(x)[2:]+'.SH' if str(x).startswith('10') else str(x)[2:]+'.SZ')
    df_free = df_free.set_index('code')

    if i < len(date_lst) - 1:
        date_new = date_lst[i+1]
        year_new, month_new = date_new[:4], date_new[4:6]
        #流通股本
        df_free_new = pd.read_csv(os.path.join('/home/largefile/public/zjwang/AShareEOD', year_new, month_new, 'ASHAREEOD.{}.csv'.format(date_new)))
        df_free_new['code'] = df_free_new['ticker'].map(lambda x:str(x)[2:]+'.SH' if str(x).startswith('10') else str(x)[2:]+'.SZ')
        df_free_new = df_free_new.set_index('code')

    #获取股指持n min的收益，最后一天的最后n min计算不了。股指不需要补全
    index_dir = os.path.join('/home/largefile/public/zjwang/000905.SH', date+'.csv')
    index_df = pd.read_csv(index_dir)
    if i < len(date_lst) - 1:
        index_dir_future = os.path.join('/home/largefile/public/zjwang/000905.SH', date_lst[i+1]+'.csv')
        index_future_df = pd.read_csv(index_dir_future)
        index_df = pd.concat([index_df, index_future_df])
    index_df['close_shift'] = index_df['close'].shift(-hold_time)
    index_df['return'] = (index_df['close_shift'] - index_df['close']) / index_df['close']

    #获取每支股票的30min超额收益率
    # print('获取每支股票的30min超额收益率')
    return_array = np.array([])
    stocks_dir = os.path.join('/home/largefile/public/1minbar', date)


    for file in os.listdir(stocks_dir):
        try:
            if file in ignore_stock_lst:
                continue

            #获得单只股票的超额收益率
            if file[7:9] == 'SH':
                is_sz = 0
                if file[:2] not in ['60', '68']:
                    ignore_stock_lst.append(file)
                    continue
            if file[7:9] == 'SZ':
                is_sz = 1
                if file[:2] not in ['00', '30']:
                    ignore_stock_lst.append(file)
                    continue
            stock_dir = os.path.join(stocks_dir, file)

            #直接从之前保存的future_df_dic获取当前的df即可，这时候已经补全
            try:
                if not future_df_dic[file].empty:
                    stock_df = future_df_dic[file]
                else:
                    stock_df = pd.read_csv(stock_dir)

                    #补全：
                    last_df = last_df_dic[file]
                    if not last_df.empty:
                        stock_df = df_complete(stock_df, df_re_sz, df_re_sh, df_free, file[:6], date, is_sz, last_df, is_last=1)
                    else:
                        stock_df = df_complete(stock_df, df_re_sz, df_re_sh, df_free, file[:6], date, is_sz)

                #更新last_df_dic：
                last_df_dic[file] = stock_df

            except:
                print('date: {} , file: {} fail in df_cur, we have ignored it!'.format(date, file))
                ignore_stock_lst.append(file)
                last_df_dic[file] = pd.DataFrame()
                continue



            if i < len(date_lst) - 1:
                stock_dir_future = os.path.join('/home/largefile/public/1minbar', date_lst[i+1], file)
                if os.path.exists(stock_dir_future):
                    try:
                        stock_df_future = pd.read_csv(stock_dir_future)
                        #补全：
                        stock_df_future = df_complete(stock_df_future, df_re_sz, df_re_sh, df_free_new, file[:6], date_new, is_sz, stock_df, is_last=1)

                        #更新future_df_dic:
                        future_df_dic[file] = stock_df_future

                        #合并：
                        stock_df = pd.concat([stock_df, stock_df_future])

                    except:
                        print('date: {} , file: {} fail in df_future, we have ignored it!'.format(date, file))
                        ignore_stock_lst.append(file)
                        future_df_dic[file] = pd.DataFrame()
                        continue

                else:
                    ignore_stock_lst.append(file)
                    future_df_dic[file] = pd.DataFrame() #不然会用到未来的数据
                    continue

            #获取收益率
            stock_df['close_shift'] = stock_df['close'].shift(-hold_time)
            stock_df['return'] = (stock_df['close_shift'] - stock_df['close']) / stock_df['close']
            
            #超额收益
            assert len(stock_df) >= 240 and len(stock_df) == len(index_df)

            res = (stock_df['return'].values - index_df['return'].values)
            stock_df['extra_return'] = res
            cur_df = stock_df[:240]


            if i == 0:
                continue
            if not os.path.exists(os.path.join('./1minbar_{}'.format(hold_time), date)):
                os.makedirs(os.path.join('./1minbar_{}'.format(hold_time), date))
            cur_df.to_csv(os.path.join('./1minbar_{}'.format(hold_time), date, file), index=None)


        except:
            print('date: {} fail in 获取股票 {} 的超额收益率'.format(date, file))
            ignore_stock_lst.append(file)
            continue
    

    ignore_stock_lst = ['689009.SH.csv']