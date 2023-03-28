#-*- coding:utf-8 –*-

# '''
# 数据是 
# 246: /home/largefile/public/net_data/stockSelect/fmBase_%s.hdf的x和60min ytrue (Rtn[thisStock].lag[60min]-Rtn[000905.SH].lag[60min]),
# /home/largefile/public/HF_factors/f12/output_factors/lastday的k线因子

# 去除涨跌停/ST/上市时间小于20个交易日的股票后,
# 计算间隔10min的x和60min ytrue的rankic、14:56的x和EOD的股票rtn-zz500 rtn的rankic

# 500的eod数据在246: /home/largefile/public/eoddata/000905.SZ.csv
# 判断/home/largefile/public/net_data/stockSelect/fmBase_%s.hdf中某一列是否为x的逻辑是
# 以"[0]"结尾且不在TradingDay;LocalTime;UpdateTime;InstrumentId;midprice[thisStock][0];volume[thisStock][0];lastpx[000905.SH][0];isST;isNoTrade;Rtn[thisStock].lag[30min];Rtn[000905.SH].lag[30min];Rtn[thisStock].lag[60min];Rtn[000905.SH].lag[60min];Rtn[thisStock].lag[120min];Rtn[000905.SH].lag[120min];Rtn[thisStock].lag[240min];Rtn[000905.SH].lag[240min];index中
# '''

import os
import re
import pandas as pd 
from collections import defaultdict
from tqdm import tqdm
import numpy as np

#一些参数
date_begin, date_end = '20211101', '20220531'   #要从目标日期前一天开始，因为要求涨停
interval = 1    #每10min取一条


#获取文档并剔除不满足条件的文档
dir_x = '/home/largefile/public/net_data/stockSelect'
date_files = [file for file in os.listdir(dir_x) if re.match(r'^fmBase_\d*.hdf',file) and date_begin <= file.split('.')[0].split('_')[1] <= date_end]
date_files.sort()

#股票前一天的收盘价函数
def get_last(df_g):
    return df_g.iloc[-1]

#股票上市时间
market_df = pd.read_csv('/home/largefile/public/eoddata/aShareOnMarket_2210.csv').loc[:, ['ts_code', 'list_date']]
market_df.columns = ['InstrumentId', 'list_date']
market_df = market_df.set_index('InstrumentId')

#交易日集合
trading_days = pd.read_csv('/home/largefile/public/zhanqi/2007-2019_trade_days.csv')['0'].to_list()

#zz500的数据
df_zz500 = pd.read_csv('/home/largefile/public/eoddata/000905.SZ.csv')
df_zz500['rtn'] = df_zz500['close'] / df_zz500['pre_close'] - 1
df_zz500.set_index('trade_date', inplace=True)


#res1：计算间隔10min的x和60min ytrue的rankic
#res2：14:56的x和EOD的股票rtn-zz500 rtn的rankic
res1, res2 = [], []


for i in tqdm(range(len(date_files))):
    date_file = date_files[i]
    dir = os.path.join(dir_x, date_file) 
    df_raw = pd.read_hdf(dir)
    df = df_raw.set_index('InstrumentId')


    #第一天跳过，获取收盘价即可
    if 'close_df' not in locals().keys():
        close_df = df.groupby('InstrumentId')[['midprice[thisStock][0]']].apply(get_last)
        close_df.columns = ['close_last_day']
        continue


    #获取日期和股票名，为了找 /home/largefile/public/HF_factors/f12/output_factors/lastday的k线因子
    date = date_file.split('.')[0].split('_')[1]


    #获取因子的col name
    x_col_name_all = list(df.columns)
    s = 'TradingDay;LocalTime;UpdateTime;InstrumentId;midprice[thisStock][0];volume[thisStock][0];lastpx[000905.SH][0];isST;isNoTrade;Rtn[thisStock].lag[30min];Rtn[000905.SH].lag[30min];Rtn[thisStock].lag[60min];Rtn[000905.SH].lag[60min];Rtn[thisStock].lag[120min];Rtn[000905.SH].lag[120min];Rtn[thisStock].lag[240min];Rtn[000905.SH].lag[240min]'
    s = s.split(';')
    x_col_name = [col for col in x_col_name_all if (col not in s and col.endswith('[0]'))]


    #获取股票的昨收
    df = df.join(close_df, on='InstrumentId', how='left')
    df.dropna(subset=['close_last_day'], inplace=True)  #去掉昨天没有收盘价的股票
    

    #获取股票涨跌停上限： 科创版和创业板注册制之后是20% 主板和创业板注册制前是10% 创业板30开头 科创版68开头 注册制时间20200824
    df['limit_percentage'] = 0.1
    if date >= '20200824':
        df.loc[df.index.str.startswith('30'), 'limit_percentage'] = 0.2
        df.loc[df.index.str.startswith('68'), 'limit_percentage'] = 0.2
    is_up = (df['midprice[thisStock][0]'] >= df['close_last_day']*(1+df['limit_percentage']))
    is_down = (df['midprice[thisStock][0]'] <= df['close_last_day']*(1-df['limit_percentage']))
    df['isLimit'] = np.logical_or(is_up, is_down)


    #获取股票是否上市小于20交易日
    date_20 = trading_days[trading_days.index(int(date)) - 20]
    less20_df = (market_df >= date_20)
    df = df.join(less20_df, on=['InstrumentId'], how='left')


    #筛选：去除涨跌停/ST/上市时间小于20个交易日的股票后
    #1:去掉ST
    df = df[df['isST']==0]
    #2:去掉上市小于20交易日
    df = df[df['list_date']==False]

    #3:去掉涨跌停的(当天涨停就去掉)
    df.sort_index(inplace=True, kind='mergesort')
    limit_df = df.groupby('InstrumentId')[['isLimit']].max()
    limit_df = limit_df[limit_df==False]
    df = df[df.index.isin(limit_df.index)]

    
    #对每个股票遍历获取/home/largefile/public/HF_factors/f12/output_factors/lastday的k线因子
    stocks = list(set(df.index))
    stocks.sort()
    array_kline = []

    ignore_stocks = []
    for stock in stocks:
        dir_kline = os.path.join('/home/largefile/public/HF_factors/f12/output_factors/lastday', date, stock+'.csv')
        if not os.path.exists(dir_kline):
            ignore_stocks.append(stock)
            continue
        df_kline = pd.read_csv(dir_kline)
        if 'x_col_kline' not in locals().keys():
            x_col_kline = list(df_kline.columns)
        array_stock = df_kline.fillna(0).values
        array_stock[np.isinf(array_stock)] == 0
        array_kline.append(array_stock)
    array_kline = np.array(array_kline)   # shape: stock_num * 240 * factor_num


    #去掉忽略的股票
    df = df[~df.index.isin(ignore_stocks)]
    df.sort_index(inplace=True, kind='mergesort')


    #因子
    array_factors = df[x_col_name].values.reshape(-1, 240, len(x_col_name))  # shape: stock_num * 240 * factor_num
    array_factors = np.concatenate([array_factors, array_kline], axis=-1)


    #获取60min y_true
    df['y_true'] = df['Rtn[thisStock].lag[30min]'] - df['Rtn[000905.SH].lag[30min]']

    #获取 EOD的股票rtn- zz500 rtn
    year, month = str(date)[:4], str(date)[4:6]
    dir_eod = os.path.join('/home/largefile/public/eoddata', year, month, 'ASHAREEOD.{}.csv'.format(date))
    df_eod = pd.read_csv(dir_eod)
    df_eod['y_true_eod'] = df_eod['ClosePrice']/df_eod['PreClosePrice'] - 1 - df_zz500.loc[int(date), 'rtn']
    df_eod['InstrumentId'] = df_eod['ticker'].map(lambda x:str(x)[2:]+'.SH' if str(x).startswith('10') else str(x)[2:]+'.SZ')
    df_eod.set_index('InstrumentId', inplace=True)
    df = df.join(df_eod['y_true_eod'], on='InstrumentId', how='left')


    #更新收盘价
    close_df = df_raw.groupby('InstrumentId')[['midprice[thisStock][0]']].apply(get_last)
    close_df.columns = ['close_last_day']


#计算rank_ic
# -----计算间隔10min的x和60min ytrue的rankic
    #因子
    array_factors_ = array_factors[:, np.arange(240//interval)*interval, :]
    rank_array_factors = array_factors_.argsort(axis=0).argsort(axis=0)
    #y_true
    y_true = df['y_true'].values.reshape(-1, 240)
    rank_y_true = y_true[:, np.arange(240//interval)*interval]
    rank_y_true = rank_y_true.argsort(axis=0).argsort(axis=0)
    #减mean + 扩展
    rank_array_factors = rank_array_factors - rank_array_factors.mean(axis=0)  # shape: stock_num * 240(/interval) * factor_num
    rank_y_true = rank_y_true - rank_y_true.mean(axis=0)                       # shape: stock_num * 240(/interval)
    rank_y_true = np.expand_dims(rank_y_true, axis=-1)
    rank_y_true = rank_y_true.repeat(rank_array_factors.shape[-1], axis=-1)    # shape: stock_num * 240(/interval) * factor_num

    #求demoninator
    demoninator = np.linalg.norm(rank_array_factors, ord=2, axis=0) * np.linalg.norm(rank_y_true, ord=2, axis=0)
    #求numerator
    numerator = np.einsum('ijk, ijk -> jk', rank_array_factors, rank_y_true)

    #得到rank_ic
    rank_ic = numerator / demoninator
    res_df = pd.DataFrame(rank_ic, columns=x_col_name + x_col_kline, index=[date]*len(rank_ic))
    res_df.index.name = 'date'
    res1.append(res_df)



# -----14:56的x和EOD的股票rtn-zz500 rtn的rankic
    #因子
    array_factors_ = array_factors[:, -5, :]
    rank_array_factors = array_factors_.argsort(axis=0)
    #y_true
    y_true = df['y_true_eod'].values.reshape(-1, 240)
    rank_y_true = y_true[:, -5]
    rank_y_true = rank_y_true.argsort(axis=0)
    #减mean + 扩展
    rank_array_factors = rank_array_factors - rank_array_factors.mean(axis=0)  # shape: stock_num *  factor_num
    rank_y_true = rank_y_true - rank_y_true.mean(axis=0)                       # shape: stock_num * 
    rank_y_true = np.expand_dims(rank_y_true, axis=-1)
    rank_y_true = rank_y_true.repeat(rank_array_factors.shape[-1], axis=-1)    # shape: stock_num * factor_num


    #求demoninator
    demoninator = np.linalg.norm(rank_array_factors, ord=2, axis=0) * np.linalg.norm(rank_y_true, ord=2, axis=0)
    #求numerator
    numerator = np.einsum('ik, ik -> k', rank_array_factors, rank_y_true)   

    #得到rank_ic
    rank_ic = (numerator / demoninator).reshape(1, -1)
    res_df = pd.DataFrame(rank_ic, columns=x_col_name + x_col_kline, index=[date]*len(rank_ic))
    res_df.index.name = 'date'
    res2.append(res_df)



res1 = pd.concat(res1)
res2 = pd.concat(res2)
res1.to_csv('./rank_ic_1min_30ytrue.csv')
res2.to_csv('./rank_ic_eod_30ytrue.csv')



    

    



    
    
    

    
    
    



    


    


