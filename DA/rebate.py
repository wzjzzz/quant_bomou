#-*- coding:utf-8 –*-


# argv[1] config路径
# UserId
# 开始时间
# 结束时间
# brokerid
# $ExchangeId.csv路径
# rebaterate

# 在*_Settlement.csv中取出UserId行，
# 根据ExchangeId寻找$ExchangeId.csv的brokerid列归类求和
# rebate=$ExchangeId.csv[product,brokerid]*Commission*rebaterate



import pandas as pd
import os
import sys
import re
from collections import defaultdict

# %%
config_dir, output_dir = sys.argv[1], sys.argv[2]
with open(config_dir, 'r') as f:
    config = f.read()

config = config.split('\n')

userid, begin_date, end_date, brokerid, exchangeid_dir, rebaterate = config[0], config[1], config[2], config[3], config[4], float(config[5])

files = os.listdir(exchangeid_dir)

settlement_file = [file for file in files if re.match('\d*_Settlement.csv', file) and begin_date <= file.split('_')[0] <= end_date]



# %%
def getInsType(insName):
    for i in range(len(insName)):
        if not ((insName[i] >= 'A' and insName[i] <= 'Z') or (insName[i] >= 'a' and insName[i] <= 'z') or insName[i] == '|'):
            return insName[:i]
    return insName

exchangeid_dic = defaultdict(set)
for file in settlement_file:
    dir = os.path.join(exchangeid_dir, file)
    df = pd.read_csv(dir)
    df = df[df.UserId == int(userid)]
    for i in range(len(df)):
        insname, Commission, exchangeid = df.iloc[i].InstrumentId, df.iloc[i].Commission, df.iloc[i].ExchangeId
        product = getInsType(insname)
        exchangeid_dic[exchangeid] |= set([(Commission, product)])


# %%
product_rebate_dic = defaultdict(float)

for exchangeid in exchangeid_dic.keys():
    dir = os.path.join(exchangeid_dir, exchangeid+'.csv')
    df = pd.read_csv(dir)
    df = df.groupby('product').sum()[brokerid]

    for Commission, product in exchangeid_dic[exchangeid]:
        num = df[product] * float(Commission) * rebaterate
        product_rebate_dic[product] += num


# %%

output_df = pd.DataFrame(product_rebate_dic.items(), columns=['product', 'brokerid'])
output_df.to_csv('./output.csv', index=False)
