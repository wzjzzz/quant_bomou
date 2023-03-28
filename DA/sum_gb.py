#-*- coding:utf-8 –*-

# argv1/home/largefile/zjwanggroup/test
# argv2月份
# 读$argv1目录下的order_cancel_stat${argv2}*_bybroker.csv
# 根据第一列归类求和统计，ProfitPreCost LivePnl LiveVlm_RMB
# 输出
# 第一列ProfitPreCost LivePnl LiveVlm_RMB


import sys
import pandas as pd
import os
import re

dir, month, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
filelst = os.listdir(dir)

pattern = 'order_cancel_stat' + month + '\d{2}_bybroker.csv'
filelst = [file for file in filelst if re.match(pattern, file)]

columns = ['ProfitPreCost','LivePnl','LiveVlm_RMB']


df_all = []
for file in filelst:
    dir_file = os.path.join(dir, file)
    df = pd.read_csv(dir_file)
    df_all.append(df)

df_all = pd.concat(df_all)

df_all = df_all.groupby('Instrument').sum()[columns]
df_sum = pd.DataFrame(df_all.sum().values.reshape(1,-1),  columns=list(df_all.columns), index=['total'])
df_all = pd.concat([df_all, df_sum])

df_all = df_all.astype(int)
df_all.reset_index(inplace=True)


df_all.to_csv(output_dir, index=False)

