#Get the column name corresponding to the minimum value of each row
import pandas as pd
import os
from functools import reduce


#res: index,  columns: df.columns
def get_maxnum_column(df, columns):
    res, max_val = [], float('-inf')
    for i in range(len(df)):
        if df[i] > max_val:
            res = [i]
            max_val = df[i]
        elif df[i] == max_val:
            res.append(i)
    return list(columns[res])


wd = '/home/largefile/zjwang' #input_data_wd
file_lst = os.listdir(wd) 
res = []
for file in file_lst:
    if file != 'res.csv':
        url = os.path.join(wd, file)
        df = pd.read_csv(url, index_col=0)
        res_tmp = df.apply(get_maxnum_column, axis=1, args=([df.columns]))
        res.append(res_tmp)

res_df = pd.concat(res)
res_url = os.path.join(wd, 'res.csv')
res_df.to_csv(res_url)
