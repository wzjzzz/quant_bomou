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

def CombineInstrumentId(prodid, year, month, nDigit):
	if (nDigit == 3):
		return prodid + ('%d' % (year % 10)) + ('%02d' % month)
	else:
		return prodid + ('%02d' % (year % 100)) + ('%02d' % month)

def ShiftInstId(baseinstid, shift):
	month = int(baseinstid[-2:]) - 1
	if baseinstid[-4].isdigit():
		year = int(baseinstid[-4:-2])
	else:
		year = int(baseinstid[-3:-2])
	month = month + shift
	while (month < 0):
		year = year - 1
		month = month + 12
	while (month >= 12):
		year = year + 1
		month = month - 12
	while year < 0:
		year = year + 100
	month = month + 1
	if baseinstid[-4].isdigit():
		return CombineInstrumentId(baseinstid[:-4], year, month, 4)
	else:
		return CombineInstrumentId(baseinstid[:-3], year, month, 3)

def get_matd(inst_month,trading_day):
	counter=0
	for i in range(1,32):
		weekday=calendar.weekday(int("20"+inst_month[:2]),int(inst_month[2:]),i)
		if weekday==4:
			mature_day=int("20"+inst_month[:2])*10000+int(inst_month[2:])*100+i
			counter+=1
			if counter==3:
				break
	if mature_day not in trading_day:
		#loc=trading_day.index(mature_day)
		#loc+=1
		#mature_day=trading_day[loc]
		mature_day=[day for day in trading_day if day >= mature_day][0]
	return str(mature_day)

def GetV1Inst(prodid):
	trading_day=pd.read_csv("/old_home/lyw/newCode2019/support/trading_days.csv")
	trading_day=list(trading_day["day"])
	res = CombineInstrumentId(prodid, tradingday.year, tradingday.month, 4)
	mature_day=get_matd(res[2:],trading_day)
	if mature_day <= str(trading_day[trading_day.index(int(sys.argv[1]))+1]):
		res=ShiftInstId(res, 1) 
	return res

def GetVNextMonthInst(prodid):
	trading_day=pd.read_csv("/old_home/lyw/newCode2019/support/trading_days.csv")
	trading_day=list(trading_day["day"])
	res = CombineInstrumentId(prodid, tradingday.year, tradingday.month, 4)
	res=ShiftInstId(res, 1)
	if res == GetV1Inst(prodid):
		res=ShiftInstId(res, 1)
	return res

def mapTradingDay(x):
	trading_day=pd.read_csv("/old_home/lyw/newCode2019/support/trading_days.csv")
	trading_day=list(trading_day["day"])
	return max([i for i in trading_day if i<=x])
def diffDate(x,y):
	trading_day=pd.read_csv("/old_home/lyw/newCode2019/support/trading_days.csv")
	trading_day=list(trading_day["day"])
	x,y=mapTradingDay(int(x)),mapTradingDay(int(y))
	return trading_day.index(y)-trading_day.index(x)
def nextMonth10(day):
	day=str(day)
	shift=1
	month = int(day[4:6]) - 1
	year = int(day[:4])
	month = month + shift
	while (month < 0):
		year = year - 1
		month = month + 12
	while (month >= 12):
		year = year + 1
		month = month - 12
	while year < 0:
		year = year + 100
	month = month + 1
	return "20"+CombineInstrumentId("IX", year, month, 4)[2:]+"10"

def GetSeasonProd(prodid):
	res = GetV1Inst(prodid)
	while (int(res[-2:]) % 3 <> 0):
		res = ShiftInstId(res, 1)
	return res

def GetNextSeasonProd(prodid):
	res = GetV1Inst(prodid)
	while (int(res[-2:]) % 3 <> 0):
		res = ShiftInstId(res, 1)
	res = ShiftInstId(res, 3)
	return res
'''
def GetVCInst(prodid):
    trading_day=pd.read_csv("/old_home/lyw/newCode2019/support/trading_days.csv")
    trading_day=list(trading_day["day"])
    res = CombineInstrumentId(prodid, tradingday.year, tradingday.month, 4)
    res=ShiftInstId(res, 1)
    if GetV1Inst(prodid) == res:
        res=ShiftInstId(res, 1)
    return res
def GetVSInst(prodid):
    res = GetV1Inst(prodid)
    res = ShiftInstId(res, 2)
    while (int(res[-2:]) % 3 <> 0):
        res = ShiftInstId(res, 1)
    return res
'''
def GetVCInst(prodid):
    trading_day=pd.read_csv("/old_home/lyw/newCode2019/support/trading_days.csv")
    trading_day=list(trading_day["day"])
    #res = CombineInstrumentId(prodid, tradingday.year, tradingday.month, 4)
    #res=ShiftInstId(res, 1)
    res=GetV1Inst(prodid)
    mature_day=get_matd(sys.argv[1][2:-2],trading_day)
    if sys.argv[1]==mature_day:
        res=ShiftInstId(res, -1)
    elif sys.argv[1]==str(trading_day[trading_day.index(int(mature_day))-1]):
        res=ShiftInstId(res, -1)
    else:
        res=ShiftInstId(res, 1)
    return res
def GetVSInst(prodid):
    res = max(GetVCInst(prodid),GetV1Inst(prodid))
    res = ShiftInstId(res, 1)
    while (int(res[-2:]) % 3 <> 0):
        res = ShiftInstId(res, 1)
    return res
