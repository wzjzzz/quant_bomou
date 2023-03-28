import pandas as pd
df = pd.read_csv('./InstrumentInfo.csv')
df = df[df['Exchange'] == 'INE']
tmp = df.ProductID.values

with open('./tcpsendinstlist.txt') as f:
    a = f.readlines()

res = [ins for ins in a if ins[:-1] in tmp]

with open('./extrainstidlist.txt', 'w') as f:
    f.writelines(res)