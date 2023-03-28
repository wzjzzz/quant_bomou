'''如果有一笔实盘委托，合约是instid，价格p，数量vol，下单的时间为localtime，
去order里找到order.localtime>localtime&&order.localtime<localtime+20ms的委托，
如果合约=instid，价格=p，数量=vol的委托只有一笔，则计算行情到委托的时间差，
否则就抛弃（20ms为可以调整的参数)
'''

'''
输入eg： 
argv[1] = 20ms
argv[2] = 行情地址
argv[3] = 实盘委托地址
argv[4] = 实盘委托column列表，对应 header_md_order
argv[5] = 输出地址
argv[6] = 是否有header
'''


import pandas as pd
import numpy as np
from functools import reduce
import sys

time = sys.argv[1]
dir1, dir2, dir3, dir4 = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
header = int(sys.argv[6])


def time_add(x, timedelta):
    x = pd.to_datetime(x)
    t = pd.Timedelta(timedelta)
    x_ = x + t
    return ''.join(str(x_).split('-'))


def get_range(line, df_market, p1, p2, lt_idx, lt_up_idx, lt_mk_idx):
    while p2 < len(df_market) and line[lt_up_idx] > df_market[p2][lt_mk_idx]:
        p2 += 1
    while p1 < p2 and line[lt_idx] >= df_market[p1][lt_mk_idx]:
        p1 += 1
    return p1-1, p2

def time_diff(t1, t2):
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)
    return (t2 - t1) / pd.Timedelta(seconds=1)



with open(dir3) as f:
    orig_columns = f.readline()
    orig_columns = orig_columns.split(',')
ignore_columns = ['SequenceNum', 'ChannelNo', 'ChannelSeq', 'OrderType', 'TradingOfficeTime']
columns = [col for col in orig_columns if col not in ignore_columns]
columns_idx = [orig_columns.index(col) for col in columns]
df_market = pd.read_csv(dir1, usecols=columns_idx) if header else pd.read_csv(dir1, usecols=columns_idx, header=None)

ins_mk_idx, lt_mk_idx, p_mk_idx, vol_mk_idx, buy_mk_idx = columns.index('InstrumentID'), columns.index('LocalTime'), columns.index('Price'), columns.index('Qty'), columns.index('Side')

df_mk_values = df_market.values



df = pd.read_csv(dir2)
df['InsertLocalTime'] = df['InsertLocalTime'].map(lambda x: eval(x))
df['LocalTime_up'] = df['InsertLocalTime'].map(lambda x: time_add(x, time))
columns = list(df.columns)
ins_idx, p_idx, vol_idx, lt_idx, lt_up_idx, buy_idx = \
    columns.index('InstrumentId'), columns.index('OrderPrice'), columns.index('OrderVol'), columns.index('InsertLocalTime'), columns.index('LocalTime_up'), columns.index('Buy')
df_values = df.values


cur_p, p1, p2 = 0, 0, 0
output = []
for cur_p in range(len(df_values)):
    line = df_values[cur_p]
    p1, p2 = get_range(line, df_mk_values, p1, p2, lt_idx, lt_up_idx, lt_mk_idx)
    # print(cur_p, p1, p2)

    instid, p, vol, t1, buy = line[ins_idx], line[p_idx], line[vol_idx], line[lt_idx], line[buy_idx]
    df_tmp = df_mk_values[p1:p2]
    c1 = df_tmp[:, ins_mk_idx] == instid
    c2 = df_tmp[:, p_mk_idx] == p
    c3 = df_tmp[:, vol_mk_idx] == vol
    c4 = df_tmp[:, buy_mk_idx] == 2 if buy == 0 else df_tmp[:, buy_mk_idx] == 1
 
    c = reduce(lambda x, y: np.logical_and(x, y), [c1, c2, c3, c4])
    if sum(c) == 1:
        t2 = df_tmp[c][:,lt_mk_idx][0]
        new_line = np.append(line, time_diff(t1, t2))
        output.append(new_line)


output_df = pd.DataFrame(output, columns=columns + ['time_diff'])
output_df = output_df[output_df.time_diff >= 0]
output_df.to_csv(dir4, index=0)
