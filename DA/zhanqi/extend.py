# -*- coding: utf-8 -*-
import os
import sys
import xlrd
import time
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

#getfile
class GetFile:
    
    def __init__(self, file_type=''):
        if isinstance(file_type, str):
            self.file_type = file_type.lower()
        else:
            self.file_type = ''
        self.short_name = []
        self.full_name = []
        self.short_name1 = []
        self.full_name1 = []

    def _filter(self, _file_type):
        if self.file_type != '':
            if _file_type == self.file_type:
                return True
            else:
                return False
        else:
            return True

    def get(self, fold_address):
        for fl in os.listdir(fold_address):
            new_dir = os.path.join(fold_address, fl)
            if os.path.isfile(new_dir) and self._filter(fl[-len(self.file_type):]):
                self.full_name.append(os.path.abspath(new_dir))
                self.short_name.append(fl)
            elif os.path.isdir(new_dir):
                self.get(new_dir)
        return self.short_name, self.full_name

    def get1(self, address, date):
        self.a,self.b=self.get(address)
        for i in range(len(self.a)):
            if self.a[i][:8]==date:
                self.short_name1.append(self.a[i]);self.full_name1.append(self.b[i])
        return self.short_name1, self.full_name1


def sendmsg(content,atlist=[]):
	print(content)

	import urllib3
	import json
	http = urllib3.PoolManager()
	encoded_data = json.dumps({"msgtype": "text", "text": {'content': content,'mentioned_list':atlist}}).encode('utf-8')
	rr = http.request(method='POST', url='https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=0f43aa2f-4a02-4552-ab47-80e2dcbb2038', body=encoded_data,headers={'Content-Type':'application/json'})
	#print('webhook response:{rr.data}')
	try:
		assert json.loads(rr.data).get('errcode') == 0
	except:
		pass


        
dic_status = {
    'gjsh': {'审批通过': 'success', '展期成功': 'success', '审批拒绝': 'failure' , '申请中': 'abnormal'},
    'gjsz': {'审批通过': 'success', '展期成功': 'success', '审批拒绝': 'failure' , '申请中': 'abnormal'},
    'zxsh': {'提出展期-办理中': 'success', '已拒绝展期': 'failure', '未展期': 'failure', '提出展期-待确认': 'abnormal'},
    'zxsz': {'提出展期-办理中': 'success', '已拒绝展期': 'failure', '未展期': 'failure', '提出展期-待确认': 'abnormal'},
    'rzrq': {'审批通过': 'success', '审批拒绝': 'failure', '7': 'abnormal', '未审批': 'abnormal'},
}


dic_msg = {
    'success': '{stock} {broker} {end_date} 展期成功 \n' + '    backup: e to {end_date} \n' + '    end: {end_date} to 99999999 \n' ,
    'failuer': '{stock} {broker} {end_date} 展期失败 \n' + '    backup: e to "" \n ',
    'abnormal': '{stock} {broker} {end_date} 审批状态异常 \n',
    'pass': '',
}


dic_column = {
    'gjsh': {'code': '证券代码', 'vol': '申请展期股数', 'date': '原到期日', 'status': '审批状态'},
    'gjsz': {'code': '证券代码', 'vol': '申请股数', 'date': '原合约到期日', 'status': '审批状态'},
    'zxsh': {'code': '="证券代码"', 'vol': '="合约数量"', 'date': '="到期日期"', 'status': '="展期标志"'},
    'zxsz': {'code': '="证券代码"', 'vol': '="合约数量"', 'date': '="到期日期"', 'status': '="展期标志"'},
    'rzrq': {'code': '="证券代码"', 'vol': '="展期数量"', 'date': '', 'status': '="状态说明"'},
}


dic_broker = {
    'gj': 'gjsh', 
    'rzrq': 'rzrq',
    'zxsz': 'zxsz',
    'zx': 'zxsh',
    'gjsz': 'gjsz',
}


def col_map(df_, broker):
# 将extend下col的格式转化为yoyaku_loan下的形式，因为有原col下是 '=""' 的形式
    df = df_.copy()
    dic = dic_column[broker]
    code_col, vol_col, date_col, status_col = dic['code'], dic['vol'], dic['date'], dic['status']
    if broker == 'gjsh':
        pass
    elif broker == 'gjsz':
        df[date_col] = df[date_col].map(lambda x: x.replace('-', ''))
    elif broker in ['rzrq']:
        fun = lambda x: x[2:-1]
        df[code_col], df[vol_col], df[status_col], df[date_col] = df[code_col].map(fun), df[vol_col].map(fun), df[status_col].map(fun), df[date_col].map(fun)
    elif broker in ['zxsh', 'zxsz']:
        fun = lambda x: x[2:-1]
        df[code_col], df[status_col], df[date_col] = df[code_col].map(fun), df[status_col].map(fun), df[date_col].map(fun)
    return df



def get_msg(df, broker, code, vol, date, loan, i):
    msg_tmp1, msg_tmp2 = '', ''
    dic = dic_column[broker]
    code_col, vol_col, date_col, status_col = dic['code'], dic['vol'], dic['date'], dic['status']
    conditon1 = df[code_col] == code
    conditon2 = df[vol_col].astype('float').astype('int') == vol
    conditon3 = df[date_col].astype('int') > date if broker != 'rzrq' else True
    df_tmp = df[(conditon1 & conditon2 & conditon3)].reset_index(drop=True)
    if df_tmp.empty:
        return 'hhh'

    for j in range(len(df_tmp)):
        status = df_tmp.loc[j][status_col]
        dic_tmp = dic_status[broker]
        if status not in dic_tmp:
            raise ValueError(f'{status} not in {broker} in dic_column')
        else:
            status_ = dic_tmp[status]
        if status_ not in dic_msg:
            raise ValueError(f'{status_} not in dic_msg')
        else:
            msg = dic_msg[status_].format(stock=code, broker=broker, end_date=date) if status_ != 'pass' else ''
            if status_ == 'abnormal':
                msg_tmp2 += msg
            else:
                if status_ == 'success':
                    loan.at[i, 'backup'], loan.at[i, 'end'] = loan.loc[i]['end'], '99999999'
                msg_tmp1 += msg
    return msg_tmp1, msg_tmp2
        


if __name__ == '__main__':
    getfile_dir = '/home/largefile/public/zhanqi/extend'
    date_lst_dir = '/home/largefile/public/zhanqi/2007-2019_trade_days.csv'
    loans_dir = '/home/largefile/public/zhanqi/yoyaku_loan.csv'


    date_file = time.strftime('%Y%m%d',time.localtime(time.time()))
    if len(sys.argv)>=2:
        date=sys.argv[1]
    else:
        date=date_file
    m, n = GetFile('xls').get1(getfile_dir, date)
    date_list=pd.read_csv(date_lst_dir, encoding='gbk',sep='\t',header=0)
    date_list=date_list['0']
    daterange=[]
    for i in range(len(date_list)):
        if date_list[i]==int(date):
            lens=date_list[i+2]-date_list[i+1]
            for j in range(lens):
                daterange.append(date_list[i+1]+j)
            break
    loanss=pd.read_csv(loans_dir,encoding='gbk',sep=',',header=0)

    message, message1 = '', ''
    repeat={}
    if len(daterange)==1:
        loan=loanss[loanss['end']==daterange[0]]
        left_loan=loanss[loanss['end']<daterange[0]]
        right_loan=loanss[loanss['end']>daterange[0]]
    else:
        loan=loanss[(loanss['end']<=daterange[-1]) & (loanss['end']>=daterange[0])]
        right_loan=loanss[loanss['end']>daterange[-1]]
        left_loan=loanss[loanss['end']<daterange[0]]
    loan=loan.reset_index(drop=True)

    
    for p in range(len(loan)):
        if loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p]) in repeat:
            if loan['vol'][p] in repeat[loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p])]:
                message += (loan['stock'][p]+loan['broker'][p]+': 错误!存在重复的展期信息\n')
            else:repeat[loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p])].append(loan['vol'][p])
        else:repeat[loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p])] = [loan['vol'][p]]


    _list=defaultdict(pd.DataFrame)
    for i in range(len(n)):
        if n[i][-15:]=='gjsh_extend.xls':
            _list['gjsh']=pd.read_excel(xlrd.open_workbook(n[i],encoding_override="gbk"),engine='xlrd',header=5, dtype='str')
        elif n[i][-15:]=='zxsz_extend.xls':
            _list['zxsz']=pd.read_csv(n[i],encoding='gbk',sep='\t',header=0, dtype='str')
        elif n[i][-15:]=='zxsh_extend.xls':
            _list['zxsh']=pd.read_csv(n[i],encoding='gbk',sep='\t',header=0, dtype='str')
        elif n[i][-15:]=='rzrq_extend.xls':
            _list['rzrq']=pd.read_csv(n[i],encoding='gbk',sep='\t',header=0, dtype='str')
        elif n[i][-15:]=='gjsz_extend.xls':
            _list['gjsz']=pd.read_excel(n[i], header=0, dtype='str')
    

    if 'gjsz' in _list:
        _list['gjsz']['申请股数'] = [int(x) for x in _list['gjsz']['申请股数']]
        a = _list['gjsz']['申请股数'].groupby([_list['gjsz']['证券代码'],_list['gjsz']['原合约到期日'],_list['gjsz']['审批状态']]).sum().reset_index()
        a = a[a['审批状态']!='已撤'].reset_index()
        _list['gjsz'] = a
    
    for i in range(len(loan)):
        if loan['backup'][i]=='e':

            stock_code = loan['stock'][i][:6]
            broker = dic_broker[loan['broker'][i]]
            vol = int(loan['vol'][i])
            
            if not _list[broker].empty:
                df = _list[broker]; df = col_map(df, broker)
                res = get_msg(df, broker, stock_code, vol, int(date), loan, i)

                if len(res) == 1:
                    message1 += str(loan['stock'][i]) + broker+ '未找到展期信息\n'
                else:
                    message += res[0]
                    message1 += res[1]


    loan=left_loan.append(loan);loan=loan.append(right_loan)
    if message:sendmsg(message)
    if message1:sendmsg(message1)
    if len(sys.argv)>=3:
        quit()
    loan.to_csv(loans_dir,index=False)