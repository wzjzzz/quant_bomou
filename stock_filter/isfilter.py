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


#复权因子
df_re_sz = pd.read_csv('/home/largefile/public/eoddata/SZAdjF.csv', index_col=0)
df_re_sh = pd.read_csv('/home/largefile/public/eoddata/SHAdjF.csv', index_col=0)
df_re_sz.fillna(1, inplace=True)
df_re_sh.fillna(1, inplace=True)


#股票上市时间
market_df = pd.read_csv('/home/largefile/public/eoddata/aShareOnMarket_2210.csv').loc[:, ['ts_code', 'list_date']]
market_df.columns = ['InstrumentId', 'list_date']
market_df = market_df.set_index('InstrumentId')


#交易日集合
trading_days = pd.read_csv('/home/largefile/public/zhanqi/2007-2019_trade_days.csv')['0'].to_list()


#股票前一天的收盘价函数
def get_last(df_g):
    return df_g.iloc[-1]


#获取日期
dir_x = '/home/largefile/public/1minbar'   
date_files = os.listdir(dir_x)
date_files = [date for date in date_files if date>= date_begin and date <= date_end]
date_files.sort()
print(date_files)


#储存昨收的字典

# 读取文件
if os.path.exists("./dics/close_lastday_dic.pkl"):
    with open("./dics/close_lastday_dic.pkl", "rb") as tf:
        close_lastday_dic = pickle.load(tf)
else:
    close_lastday_dic = defaultdict(float)
#储存过去20天的AMT(前20天不太置信)
if os.path.exists("./dics/AMT_mean_dic.pkl"):
    with open("./dics/AMT_mean_dic.pkl", 'rb') as tf:
        AMT_mean_dic = pickle.load(tf)
else:
    AMT_mean_dic = defaultdict(deque)
#储存昨天的数据，用于补全
if os.path.exists("./dics/dic.pkl"):
    with open("./dics/dic.pkl", 'rb') as tf:
        dic = pickle.load(tf)
else:
    dic = defaultdict(pd.DataFrame)


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
    #储存每天涨停的结果
    res_limit = []
    
    #储存每天最近20天上市日期或日均成交额过低
    res_list_dic, res_AMT_dic = {}, {}

    date = date_files[i]
    year, month = date[:4], date[4:6]
    dir = os.path.join(dir_x, date)
    stocks = os.listdir(dir)
    df_eod = pd.read_csv(os.path.join('/home/largefile/public/eoddata', year, month, 'ASHAREEOD.{}.csv'.format(date)), index_col=0)
    for stock in stocks:
        try:
            #获取一些信息
            stock_name = stock[:-4]
            if stock[7:9] == 'SH':
                is_sz = 0
                ticker = '10' + stock[:6]
                if stock[:2] not in ['60', '68']:
                    continue
            if stock[7:9] == 'SZ':
                is_sz = 1
                ticker = '11' + stock[:6]
                if stock[:2] not in ['00', '30']:
                    continue
            df = pd.read_csv(os.path.join(dir, stock))
            try:
                amount = df_eod.loc[int(ticker), 'AMT']
            except:
                amount = df_eod.loc[ticker, 'AMT']

            if stock_name not in close_lastday_dic or stock_name not in AMT_mean_dic or stock not in dic:
                AMT_mean_dic[stock_name] = AMT_process(AMT_mean_dic[stock_name], amount)
                df = df_complete(df, df_re_sz, df_re_sh, stock[:6], date, is_sz)
                dic[stock] = df
                close_lastday_dic[stock_name] = df['close'].iloc[-1]
                continue
            df_last = dic[stock]
            df = df_complete(df, df_re_sz, df_re_sh, stock[:6], date, is_sz, df_last, is_last=1)

            #判断是否涨停：科创版和创业板注册制之后是20% 主板和创业板注册制前是10% 创业板30开头 科创版68开头
            limit_percentage = 0.1
            if stock.startswith('30') or stock.startswith('68'):
                limit_percentage = 0.2
            is_up = df['close'] >= (limit_percentage + 1)*close_lastday_dic[stock_name]
            is_down = df['close'] <= (1 - limit_percentage)*close_lastday_dic[stock_name]
            df[stock_name] = np.logical_or(is_up, is_down) + 0   #+0 为了转化为int


            #获取股票是否上市小于等于20交易日
            islist = 0
            date_20 = trading_days[trading_days.index(int(date)) - 20]
            list_date = market_df.loc[stock_name, 'list_date']
            if list_date >= date_20:
                islist = 1
            res_list_dic[stock_name] = islist
            

            #日成交金额均值低于2kw
            isAMT = 0
            if np.mean(AMT_mean_dic[stock_name]) <= amount_limit:
                isAMT = 1
            res_AMT_dic[stock_name] = isAMT
            
            #更新收盘价和历史成交金额和昨天的df
            close_lastday_dic[stock_name] = df['close'].iloc[-1]
            AMT_mean_dic[stock_name] = AMT_process(AMT_mean_dic[stock_name], amount)
            dic[stock] = df


            df = df.loc[:, [stock_name]]
            df.reset_index(inplace=True, drop=True)
            df.index.name = 'seqNo'
            res_limit.append(df)

        except:
            print(f'{date}-{stock}-failed!!')
        
    if i > 0 or os.path.exists("./dics/close_lastday_dic.pkl"):
        res_list.append(res_list_dic)
        res_AMT.append(res_AMT_dic)
        res_limit = pd.concat(res_limit, axis=1)
        res_limit.to_csv('/home/largefile/public/stock_filter/results/limit_minbar_daily/limit_{}.csv'.format(date))
        

res_list = pd.DataFrame(res_list, index=date_files[1:])
res_list.index.name = 'date'
res_AMT = pd.DataFrame(res_AMT, index=date_files[1:])
res_AMT.index.name = 'date'

def merge(dir, df):
    if os.path.exists(dir):
        df_exist = pd.read_csv(dir, index_col=0)
        df_exist.to_csv( os.path.join('/'.join(dir.split('/')[:-1]), 'old_' + dir.split('/')[-1]))  #保存原始的以防出问题
        df_exist = df_exist.T
        df_exist.index.name = 'stock'
        df = df.T
        df.index.name = 'stock'
        df_exist = df_exist.merge(df, on='stock', how='outer')
        df_exist.fillna(0, inplace=True)
        df_exist.index.name = None
        return df_exist.T
    else:
        return df

dir_list = '/home/largefile/public/stock_filter/results/list_less20.csv'
dir_AMT = '/home/largefile/public/stock_filter/results/amt_less_2.csv'
res_list = merge(dir_list, res_list)
res_AMT = merge(dir_AMT, res_AMT)
res_list.to_csv(dir_list)
res_AMT.to_csv(dir_AMT)
    

        

# ===============保存字典    
with open("./dics/close_lastday_dic.pkl", "wb") as tf:
    pickle.dump(close_lastday_dic, tf)

with open("./dics/AMT_mean_dic.pkl", 'wb') as tf:
    pickle.dump(AMT_mean_dic, tf)

with open("./dics/dic.pkl", 'wb') as tf:
    pickle.dump(dic, tf)






