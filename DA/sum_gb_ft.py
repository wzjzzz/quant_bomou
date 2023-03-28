#-*- coding:utf-8 –*-

# argv1/home/largefile/zjwanggroup/test
# argv2月份
# 月份*_Settlement.csv
# 根据ExchangeId归类求和TradeVolume,Profit,Commission,NetProfit

import sys
import pandas as pd
import os
import re

dir, month, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
filelst = os.listdir(dir)

pattern = month + '\d{2}_Settlement.csv'
filelst = [file for file in filelst if re.match(pattern, file)]

columns = ['TradeVolume', 'Profit', 'Commission', 'NetProfit']


df_all = []
for file in filelst:
    dir_file = os.path.join(dir, file)
    df = pd.read_csv(dir_file)
    df_all.append(df)

df_all = pd.concat(df_all)

df_all = df_all.groupby('ExchangeId').sum()[columns]
df_all = df_all.astype(int)
df_all.reset_index(inplace=True)


df_all.to_csv(output_dir, index=False)

