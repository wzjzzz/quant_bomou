import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import os
from collections import defaultdict
import sys


# %%
def df_process(df1):
    '''用close列计算1min后收益率 '''

    df = df1.copy()
    df['close_shift'] = df['close'].shift(1)
    # df.loc[0, 'close_shift'] = df.loc[0,'open']
    df['return'] = (df['close'] - df['close_shift']) / df['close_shift']
    return df


def df_complete(df, df_re_sz, df_re_sh, df_free, inst, date, is_sz, last_df=None, is_last=0):
    '''股票df完善:
        1、中间部分min没有成交记录的用前1min补充, 如果是刚开始就没成交记录的用前一天的最后一条补充; (由于原始数据已经补充了中间的, 这里只需补充开始和结尾没有成交记录的)
        2、复权因子处理 + 计算换手率
    '''

    #复权因子处理 + 计算换手率
    if is_sz:
        df['close'] = df['close'] * df_re_sz.loc[int(inst), date] 
        df['turnover_rate'] = df['volume'] / df_free.loc[inst+'.SZ', 'FLOAT_A_SHR']   
    else:
        df['close'] = df['close'] * df_re_sh.loc[int(inst), date]
        df['turnover_rate'] = df['volume'] / df_free.loc[inst+'.SH', 'FLOAT_A_SHR']   


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
    return df
        



# %%
# 回看
# #存在nan
#服务器的python版本好像不支持 rolling(closed='left')，必须要datelike index
def factors_gen1(df):
    '''
    1)高频收益方差/偏度/峰度 
    2)上行波动 下行波动 和占比 
    3)资金流因子 对每只股票求各自的x
    4)反转因子 wi可以取volume^0.5和alpha*tanh(volume/alpha), alpha是常数
    以上都测试不同回看时间(历史10min/30min/60min)和是否使用前一天的收益率
    '''

    intervals = [10, 30, 60]

    factors_all = []

    for interval in intervals:

        # var = df['return'].groupby(lambda x: math.floor(x/interval)).var()
        # skew = df['return'].groupby(lambda x: math.floor(x/interval)).skew()
        # kurt = df['return'].groupby(lambda x: math.floor(x/interval)).apply(pd.DataFrame.kurt)
        '''分钟级别数据——高频收益方差/偏度/峰度  1、(1)(2)(3)   对应列名: var_{}, skew_{}, kurt_{}  {}内为回看时间, 10/30/60'''
        var = df['return'].apply(lambda x: x**2).rolling(interval, min_periods=0).sum()
        var[var<1e-6] = 1e-6
        skew = np.sqrt(interval) * df['return'].apply(lambda x: x**3).rolling(interval, min_periods=0).sum() / var**1.5
        kurt = interval * df['return'].apply(lambda x: x**4).rolling(interval, min_periods=0).sum() / var**2 - 3

        #----------------------------
        '''分钟级别数据——上行/下行波动  2、(8)(9)  对应列名: up_fluctuate_{}, down_fluctuate_{}   {}为回看时间， 10/30/60'''
        tmp_df = df.copy()
        tmp_df[tmp_df['return'] <=0 ] = 0
        up_fluctuate = tmp_df['return'].apply(lambda x: x**2).rolling(interval, min_periods=0).sum()**0.5
        # up_fluctuate.fillna(0, replace=True)
        amount_in = tmp_df['turnover'].rolling(interval, min_periods=0).sum()

        tmp_df = df.copy()
        tmp_df[tmp_df['return'] >=0 ] = 0
        down_fluctuate = tmp_df['return'].apply(lambda x: x**2).rolling(interval, min_periods=0).sum()**0.5
        # down_fluctuate.fillna(0, replace=True)
        amount_out = tmp_df['turnover'].rolling(interval, min_periods=0).sum()

        #------
        '''分钟级别数据——上行/下行波动占比  2、(10)(11) 对应列名: up_fluctuate_ratio_{}, down_fluctuate_ratio_{}  {}内为回看时间， 10/30/60'''
        up_fluctuate_ratio = up_fluctuate**2 / var
        down_fluctuate_ratio = down_fluctuate**2 / var
        amount = df['turnover'].rolling(interval, min_periods=0).sum()
        '''分钟级别数据——资金流因子    3、(15)<1>  对应列名: flow_in_ratio_{}  {}内为回看时间， 10/30/60'''
        flow_in_ratio = (amount_in - amount_out) / amount


        #----------------------------------------
        #weight: sqrt(volumn)
        '''分钟级别数据——反转因子   3、(17)  wi可以取volume^0.5和alpha*tanh(volume/alpha), alpha是常数;  volume可以用换手率替代
         对应列名 rev_sqrt_{}, rev_tanh_{}, rev_sqrt_turnover{}, rev_tanh_turnover{}   {}内为回看时间， 10/30/60'''
        weight_normal = (df['volume'] ** 0.5).rolling(interval, min_periods=0).sum()
        weight = (df['volume'] ** 0.5)
        rev_sqrt = ((np.log(df['return'] + 1)) * weight).rolling(interval, min_periods=0).sum() / weight_normal

        #weight: alpha*tanh(volumn/alpha)
        alpha = 1e8
        weight_normal = (alpha*np.tanh(df['volume']/alpha)).rolling(interval, min_periods=0).sum()
        weight = alpha*np.tanh(df['volume']/alpha)
        rev_tanh = (np.log(df['return'] + 1) * weight).rolling(interval, min_periods=0).sum() / weight_normal


        #turnover version ------------------------
        #weight: sqrt(volumn)
        weight_normal = (df['turnover_rate'] ** 0.5).rolling(interval, min_periods=0).sum()
        weight = (df['turnover_rate'] ** 0.5)
        rev_sqrt_turnover = ((np.log(df['return'] + 1)) * weight).rolling(interval, min_periods=0).sum() / weight_normal

        #weight: alpha*tanh(volumn/alpha)
        alpha = 1e8
        weight_normal = (alpha*np.tanh(df['turnover_rate']/alpha)).rolling(interval, min_periods=0).sum()
        weight = alpha*np.tanh(df['turnover_rate']/alpha)
        rev_tanh_turnover = (np.log(df['return'] + 1) * weight).rolling(interval, min_periods=0).sum() / weight_normal
        

        data_lst = [var, skew, kurt, up_fluctuate, down_fluctuate, up_fluctuate_ratio, down_fluctuate_ratio, flow_in_ratio, \
                            rev_sqrt, rev_tanh, rev_sqrt_turnover, rev_tanh_turnover]
        column_lst = ["var", "skew", "kurt", "up_fluctuate", "down_fluctuate", \
            "up_fluctuate_ratio", "down_fluctuate_ratio", "flow_in_ratio", "rev_sqrt", "rev_tanh", 'rev_sqrt_turnover', 'rev_tanh_turnover']

        column_lst = [column + '_{}'.format(interval) for column in column_lst]
        factors = pd.concat(data_lst, axis=1)[-240:]
        factors.columns = column_lst
        factors_all.append(factors)
    
    factors_all = pd.concat(factors_all, axis=1)
    
    return factors_all



# %%
def factors_gen2(df):
    '''
    趋势强度, 回看10min/30min/60min/240min
    高频量价相关性, 回看30min/60min/240min
    日内成交量占比, 过去10min/30min/60min在过去60min/240min的占比
    链接中的情绪因子也生成一下
    '''
    cols = []

    '''分钟级别数据——趋势强度  3、(16)  对应列名: trendStrength_{}  {}内为回看时间,  10/30/60/240'''
    intervals = [10, 30, 60, 240]
    for interval in intervals:
        denominator = abs((df['close'].shift(1) - df['close'])).rolling(interval, min_periods=0).sum()
        numerator = (df['close'].shift(interval) - df['close'])
        df['trendStrength_{}'.format(interval)] = numerator / denominator
        cols.append('trendStrength_{}'.format(interval))


    '''分钟级别数据——高频量价相关性： 3、(13)   对应列名: price_volume_corr_{}   {}内为回看时间, 30/60/240'''
    intervals = [30, 60, 240]
    for interval in intervals:
        df['price_volume_corr_{}'.format(interval)] = df['close'].rolling(interval, min_periods=0).corr(df['volume'])
        cols.append('price_volume_corr_{}'.format(interval))


    '''分钟级别数据——日内成交量占比： 3、(12)  对应列名: volume_ratio_{1}_{2}   {1}为过去时间分子 / {2}为过去时间分母   {1}: 10/30/60; {2}: 60/240   其中60/60没意义'''
    intervals = [10, 30, 60]
    _intervals = [60, 240]
    for interval1 in intervals:
        for interval2 in _intervals:
            denominator = df['volume'].rolling(interval2, min_periods=0).sum()
            numerator = df['volume'].rolling(interval1, min_periods=0).sum()
            df['volume_ratio_{}_{}'.format(interval1, interval2)] = numerator / denominator - (interval1/interval2)
            cols.append('volume_ratio_{}_{}'.format(interval1, interval2))


    #情绪因子：
    # intervals = [240]
    # df['smart_degree'] = abs(df['return']) / np.sqrt(df['volume'])
    # df['volume_price'] = df['close'] * df['volume']
    # def smart(x, df, interval):
    #     index = x.index
    #     tmp_df = df.loc[index]
    #     tmp_df = tmp_df.sort_values(by='smart_degree').cumsum()
    #     all = tmp_df.iloc[-1]['volume_price']
        
    #     threshold = tmp_df.iloc[-1]['volume'] * 0.2
    #     smart_all = tmp_df[tmp_df['volume'] >= threshold].iloc[0]['volume_price']

    #     return smart_all/all

    # for interval in intervals:
    #     df['smart_mood_{}'.format(interval)] = df['smart_degree'].rolling(interval, min_periods=0).apply(smart, args=((df, interval)))
    #     cols.append('smart_mood_{}'.format(interval))

    return df[cols][-240:]







# %%

#复权因子
df_re_sz = pd.read_csv('/home/largefile/public/eoddata/SZAdjF.csv', index_col=0)
df_re_sh = pd.read_csv('/home/largefile/public/eoddata/SHAdjF.csv', index_col=0)
df_re_sz.fillna(1, inplace=True)
df_re_sh.fillna(1, inplace=True)


date_begin = sys.argv[1]
date_end = sys.argv[2]

date_lst = os.listdir('/home/largefile/public/1minbar')
date_lst.sort()
date_lst = [date for date in date_lst if date_begin <= date <= date_end]


print(date_lst)
last_day_dfs = defaultdict(pd.DataFrame)  #store last day dfs, key:filename  value: df

for date in tqdm(date_lst):
    date_dir = os.path.join('/home/largefile/public/1minbar', date)
    filelist = os.listdir(date_dir)

    year, month = date[:4], date[4:6]
    #流通股本
    df_free = pd.read_csv(os.path.join('/home/largefile/public/AShareEOD', year, month, 'ASHAREEOD.{}.csv'.format(date)))
    df_free['code'] = df_free['ticker'].map(lambda x:str(x)[2:]+'.SH' if str(x).startswith('10') else str(x)[2:]+'.SZ')
    df_free = df_free.set_index('code')
    


    #no last day
    # output_dir = os.path.join('output_factors', 'no_lastday' , '{}min'.format(interval) , date)
    #use last day
    output_dir_last = os.path.join('./output_factors', 'lastday1' , date)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    if not os.path.exists(output_dir_last):
        os.makedirs(output_dir_last)
    for file in filelist:
        try:
            if file[7:9] == 'SH':
                is_sz = 0
                if file[:2] not in ['60', '68']:
                    continue
            if file[7:9] == 'SZ':
                is_sz = 1
                if file[:2] not in ['00', '30']:
                    continue
            dir = os.path.join('/home/largefile/public/1minbar', date, file)
            df = pd.read_csv(dir)
            df_no = df.copy()

            #no last day
            # if len(df) != 240:
            #     df_no = df_complete(df_no)
            # df1 = df_process(df_no)
            # factors_df = factors_gen(df1, interval)
            # output_dir_ = os.path.join(output_dir,file)
            # factors_df.to_csv(output_dir_, index=0)

            #last day
            last_df = last_day_dfs[file]
            try:
                if not last_df.empty:
                    df = df_complete(df, df_re_sz, df_re_sh, df_free, file[:6], date, is_sz, last_df, is_last=1)
                    last_df = df[-240:]
                    last_day_dfs[file] = last_df    #更新last_df
                else:
                    df = df_complete(df, df_re_sz, df_re_sh, df_free, file[:6], date, is_sz)
                    last_df = df[-240:]
                    last_day_dfs[file] = last_df    #更新last_df
                    continue
            except:
                print('date: {} , file: {} fail in df_complete'.format(date, file))
                continue
            

            # tmp_dir = os.path.join('./df', output_dir_last)
            # if not os.path.exists(tmp_dir):
            #     os.makedirs(tmp_dir)
            # last_df.to_csv(os.path.join('./df', output_dir_last, file))

            df1 = df_process(df)


            try:
                factors_df_last1 = factors_gen1(df1)
                factors_df_last2 = factors_gen2(df1)
                factors_df_last = pd.concat([factors_df_last1, factors_df_last2], axis=1).fillna(0)
            except:
                print('date: {} , file: {} fail in factors_gen'.format(date, file))
                continue

            output_dir_last_ = os.path.join(output_dir_last,file)


            # if date == date_lst[date_lst.index('20220802')-1]:
            #     continue

            #保存csv
            factors_df_last.to_csv(output_dir_last_, index=0)


        except:
            print('date: {} , file: {} failed!!!!'.format(date, file))


