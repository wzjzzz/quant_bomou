# -*- coding: utf-8 -*-

# 回测换月规则：
# 1、每个月第三个周五（假如是非交易日顺延到下一个交易日）的前一个交易日以后（包括当天）下月合约被认为是主力vr，否则本月合约是vr；
# 2、vc逻辑是第三个周五（非交易日顺延）的前一个交易日和本交易日是vr-1，否则vr+1；
# 3、vs逻辑是vc后的能被3整除的月份。
# 输入日期，对/home/largefile/zjwanggroup/ix/
# simulate_IXypre.txt.template里面的tmpv1，tmpvc和tmpvs替换成正确月份
# 比如日期20221019效果是
# sed 's/tmpv1/2211/g' simulate_IXypre.txt.template > simulate_IXypre.txt
# sed -i 's/tmpvc/2212/g' simulate_IXypre.txt
# sed -i 's/tmpvs/2303/g' simulate_IXypre.txt

import os
import sys
import json
import pandas as pd
import datetime
import calendar
from collections import OrderedDict
import types
import copy
import random

tradingday=datetime.datetime.strptime(sys.argv[1],"%Y%m%d")

# trading_day = [20221100 + i for i in range(1, 32)]

def ShiftInstId(inst_month, shift):
    month = int(inst_month[4:])
    year = int(inst_month[:4])
    month = month + shift
    while (month <= 0):
        year = year - 1
        month = month + 12
    while (month > 12):
        year = year + 1
        month = month - 12
    if month < 10:
        month = '0' + str(month)
    return str(year) + str(month)



def get_matd(inst_month,trading_day):
    counter=0
    for i in range(1,32):
        weekday=calendar.weekday(int(inst_month[:4]),int(inst_month[4:]),i)
        if weekday==4:
            counter+=1
            mature_day=int(inst_month[:4])*10000+int(inst_month[4:])*100+i
            if counter==3:
                break
    if mature_day not in trading_day:
        #loc=trading_day.index(mature_day)
        #loc+=1
        #mature_day=trading_day[loc]
        mature_day=[day for day in trading_day if day >= mature_day][0]
    return str(mature_day)


def GetV1Inst():
    trading_day=pd.read_csv("/home/largefile/zjwanggroup/ix/trading_days.csv")
    trading_day=list(trading_day.values.reshape(-1))
    res = str(tradingday.year) + str(tradingday.month)
    res = ShiftInstId(res, 0)
    mature_day=get_matd(res,trading_day)
    if sys.argv[1] >= str(trading_day[trading_day.index(int(mature_day))-1]):
        res=ShiftInstId(res, 1) 
    return res



def GetVCInst():
    trading_day=pd.read_csv("/home/largefile/zjwanggroup/ix/trading_days.csv")
    trading_day=list(trading_day.values.reshape(-1))
    V1=GetV1Inst()
    res = str(tradingday.year) + str(tradingday.month)
    mature_day=get_matd(res,trading_day)
    if sys.argv[1]==mature_day:
        VC=ShiftInstId(V1, -1)
    elif sys.argv[1]==str(trading_day[trading_day.index(int(mature_day))-1]):
        VC=ShiftInstId(V1, -1)
    else:
        VC=ShiftInstId(V1, 1)
    return VC


def GetVSInst():
    res = max(GetVCInst(),GetV1Inst())
    res = ShiftInstId(res, 1)
    while (int(res[4:]) % 3 != 0):
        res = ShiftInstId(res, 1)
    return res


V1, VC, VS = GetV1Inst(), GetVCInst(), GetVSInst()

# print(V1, VC, VS)

file_dir = sys.argv[3]

with open(file_dir) as f:
    res = f.read()
res = res.replace('tmpv1', V1[2:])
res = res.replace('tmpvc', VC[2:])
res = res.replace('tmpvs', VS[2:])


output_dir = sys.argv[2]
with open(output_dir, 'w') as f:
    f.write(res)


