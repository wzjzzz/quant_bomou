# 是否ST 在 246: /home/largefile/public/ashareIsST_1801_2209.csv
# 生成每个股票是否涨停的数据 
# 去除涨跌停/ST/上市时间小于20个交易日
# 去除过去20AMT日成交金额均值低于2千万


#输出结果：1表示涨停，小于20交易日，低于2kw


import os
import pandas as pd
from collections import defaultdict, deque
import numpy as np
from tqdm import tqdm
import sys
import pickle



#一些参数
amount_limit = 2e7    # 日成交金额是否低于2kw
list_day = 20       #上市时间限制
date_begin = sys.argv[1]
date_end = sys.argv[2]


#股票上市时间
market_df = pd.read_csv('/home/largefile/public/eoddata/aShareOnMarket_2210.csv').loc[:, ['ts_code', 'list_date']]
market_df.columns = ['InstrumentId', 'list_date']
market_df = market_df.set_index('InstrumentId')


#交易日集合
trading_days = pd.read_csv('/home/largefile/public/zhanqi/2007-2019_trade_days.csv')['0'].to_list()


#获取日期
date_files = [str(date) for date in trading_days if str(date) >= date_begin and str(date) <= date_end]
date_files.sort()
print(date_files)


#储存过去20天的AMT(前20天不太置信)
AMT_mean_dic = defaultdict(deque)


#保存最近的20天成交额数据
def AMT_process(dq, amount):
    dq.append(amount)
    if len(dq) > 20:
        dq.popleft()
    return dq


#储存每天最近20天上市日期或日均成交额过低        
res_list = []
res_AMT = []
for i in tqdm(range(len(date_files))):  
    #储存每天最近20天上市日期或日均成交额过低
    res_list_dic, res_AMT_dic = {}, {}

    date = date_files[i]
    year, month = date[:4], date[4:6]
    df_eod = pd.read_csv(os.path.join('/home/largefile/public/eoddata', year, month, 'ASHAREEOD.{}.csv'.format(date)))
    df_eod['code'] = df_eod['ticker'].map(lambda x:str(x)[2:]+'.SH' if str(x).startswith('10') else str(x)[2:]+'.SZ')
    df_eod.set_index('code', drop=True, inplace=True)
    stocks = list(df_eod.index)
    for stock in stocks:
        # try:
            #获取一些信息
        if stock[7:9] == 'SH':
            is_sz = 0
            if stock[:2] not in ['60', '68']:
                continue
        if stock[7:9] == 'SZ':
            is_sz = 1
            if stock[:2] not in ['00', '30']:
                continue
        try:
            amount = df_eod.loc[stock, 'AMT']
        except:
            amount = df_eod.loc[stock, 'AMT']

        if stock not in AMT_mean_dic:
            AMT_mean_dic[stock] = AMT_process(AMT_mean_dic[stock], amount)
            continue

        #获取股票是否上市小于等于20交易日
        islist = 0
        date_20 = trading_days[trading_days.index(int(date)) - 20]
        list_date = market_df.loc[stock, 'list_date']
        if list_date >= date_20:
            islist = 1
        res_list_dic[stock] = islist
        

        #日成交金额均值低于2kw
        isAMT = 0
        if np.mean(AMT_mean_dic[stock]) <= amount_limit:
            isAMT = 1
        res_AMT_dic[stock] = isAMT
        
        #历史成交金额
        AMT_mean_dic[stock] = AMT_process(AMT_mean_dic[stock], amount)

        # except:
        #     print(f'{date}-{stock}-failed!!')
        
    if i > 0:
        res_list.append(res_list_dic)
        res_AMT.append(res_AMT_dic)

        

res_list = pd.DataFrame(res_list, index=date_files[1:]).fillna(0)
res_list.index.name = 'date'
res_AMT = pd.DataFrame(res_AMT, index=date_files[1:]).fillna(0)
res_AMT.index.name = 'date'

# def merge(dir, df):
#     if os.path.exists(dir):
#         df_exist = pd.read_csv(dir, index_col=0)
#         df_exist.to_csv( os.path.join('/'.join(dir.split('/')[:-1]), 'old_' + dir.split('/')[-1]))  #保存原始的以防出问题
#         df_exist = df_exist.T
#         df_exist.index.name = 'stock'
#         df = df.T
#         df.index.name = 'stock'
#         df_exist = df_exist.merge(df, on='stock', how='outer')
#         df_exist.fillna(0, inplace=True)
#         df_exist.index.name = None
#         return df_exist.T
#     else:
#         return df


dir_list = '/home/largefile/public/111/results/list_less20.csv'
dir_AMT = '/home/largefile/public/111/results/amt_less_2.csv'
# res_list = merge(dir_list, res_list)
# res_AMT = merge(dir_AMT, res_AMT)
res_list.to_csv(dir_list)
res_AMT.to_csv(dir_AMT)
    

        








