# -*- coding: utf-8 -*-
import os
import sys
import xlrd
import time
import pandas as pd
import numpy as np
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

    def get1(self,address):
        self.a,self.b=self.get(address)
        for i in range(len(self.a)):
            if self.a[i][:8]==date_file:
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
 
if __name__ == '__main__':
    date_file = time.strftime('%Y%m%d',time.localtime(time.time()))
    if len(sys.argv)>=2:
        date=sys.argv[1]
    else:
        date=date_file
    m, n = GetFile('xls').get1('/home/largefile/ftao/extend')
    date_list=pd.read_csv('/old_home/data/public/intern/zrsun/2007-2019_trade_days.csv',encoding='gbk',sep='\t',header=0)
    date_list=date_list['0']
    daterange=[]
    for i in range(len(date_list)):
        if date_list[i]==int(date):
            lens=date_list[i+2]-date_list[i+1]
            for j in range(lens):
                daterange.append(date_list[i+1]+j)
            break
    loanss=pd.read_csv('/home/largefile/ftao/yoyaku_loan.csv',encoding='gbk',sep=',',header=0)
    #loanss=pd.read_csv('/home/largefile/ftao/yoyaku_bydate//yoyaku_loan_20221124.csv',encoding='gbk',sep=',',header=0)
    message=[];message1=[]
    repeat={}
    if len(daterange)==1:
        loan=loanss[loanss['end']==daterange[0]];left_loan=loanss[loanss['end']<daterange[0]];right_loan=loanss[loanss['end']>daterange[0]]
    else:loans=loanss[loanss['end']<=daterange[-1]];loan=loans[loans['end']>=daterange[0]];right_loan=loanss[loanss['end']>daterange[-1]];left_loan=loanss[loanss['end']<daterange[0]]
    loan=loan.reset_index(drop=True)
    #os.mkdir(os.path.join('/old_home/data/public/intern/zrsun/yoyaku',date))
    #loan.to_csv(os.path.join('/old_home/data/public/intern/zrsun/yoyaku',date,date+'split.csv'),index=False)
    
    for p in range(len(loan)):
        if loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p]) in repeat:
            if loan['vol'][p] in repeat[loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p])]:
                message.append(loan['stock'][p]+loan['broker'][p]+': 错误!存在重复的展期信息\n')
            else:repeat[loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p])].append(loan['vol'][p])
        else:repeat[loan['stock'][p]+loan['broker'][p]+str(loan['start'][p])+str(loan['end'][p])]=[loan['vol'][p]]
    _list={};a=0
    for i in range(len(n)):
        if n[i][-15:]=='gjsh_extend.xls':
            _list['gjsh']=pd.read_excel(xlrd.open_workbook(n[i],encoding_override="gbk"),engine='xlrd',header=5)
        elif n[i][-15:]=='zxsz_extend.xls':
            _list['zxsz']=pd.read_csv(n[i],encoding='gbk',sep='\t',header=0)
        elif n[i][-15:]=='zxsh_extend.xls':
            _list['zxsh']=pd.read_csv(n[i],encoding='gbk',sep='\t',header=0)
        elif n[i][-15:]=='rzrq_extend.xls':
            _list['rzrq']=pd.read_csv(n[i],encoding='gbk',sep='\t',header=0)
        elif n[i][-15:]=='gjsz_extend.xls':
            _list['gjsz']=pd.read_excel(xlrd.open_workbook(n[i],encoding_override="gbk"),engine='xlrd',header=0)
    
    if 'gjsz' in _list:
        _list['gjsz']['申请股数'] = [int(x) for x in _list['gjsz']['申请股数']]
        a = _list['gjsz']['申请股数'].groupby([_list['gjsz']['证券代码'],_list['gjsz']['原合约到期日'],_list['gjsz']['审批状态']]).sum().reset_index()
        a=a[a['审批状态']!='已撤'].reset_index()
        _list['gjsz'] = a
    
    for i in range(len(loan)):
        if loan['backup'][i]=='e':
            count=0
            stock_code=loan['stock'][i][:6]
            broker=loan['broker'][i]
            vol=int(loan['vol'][i])
            if broker=='gj':
                if 'gjsh' in _list:
                    for j in range(len(_list['gjsh'])):
                        if str(_list['gjsh']['证券代码'][j])==stock_code and int(_list['gjsh']['申请展期股数'][j])==vol and int(_list['gjsh']['原到期日'][j])>int(date):
                            count=1
                            if _list['gjsh']['审批状态'][j]=='审批同意':message.append(loan['stock'][i]+'gjsh展期成功\n     backup: e to '+str(loan['end'][i])+' \n     end: '+str(loan['end'][i])+' to 99999999\n');loan['backup'][i]=loan['end'][i];loan['end'][i]='99999999'
                            elif _list['gjsh']['审批状态'][j]=='未审批':continue
                            elif _list['gjsh']['审批状态'][j]=='审批拒绝':message.append(loan['stock'][i]+'gjsh展期失败\n     backup: e to “ ”\n');loan['backup'][i]=np.nan
                            else:message1.append(str(loan['stock'][i])+'gjsh,审批状态异常\n')
                            break
            elif broker=='rzrq':
                if 'rzrq' in _list:
                    for k in range(len(_list['rzrq'])):
                        if str(_list['rzrq']['="证券代码"'][k][-7:-1])==stock_code and int(_list['rzrq']['="展期数量"'][k][2:-1])==vol:
                            count=1
                            if _list['rzrq']['="状态说明"'][k][-5:-1]=='审批通过':message.append(loan['stock'][i]+'rzrq展期成功\n     backup: e to '+str(loan['end'][i])+' \n     end: '+str(loan['end'][i])+' to 99999999\n');loan['backup'][i]=loan['end'][i];loan['end'][i]='99999999'
                            elif _list['rzrq']['="状态说明"'][k][-5:-1]=='审批拒绝':message.append(loan['stock'][i]+'rzrq展期失败\n     backup: e to “ ”\n');loan['backup'][i]=np.nan
                            elif _list['rzrq']['="状态说明"'][k][-2:-1]=='7':continue
                            else:message1.append(str(loan['stock'][i])+'rzrq,审批状态异常\n')
                            break
            elif broker=='zxsz':
                if 'zxsz' in _list:
                    for h in range(len(_list['zxsz'])):
                        if str(_list['zxsz']['="证券代码"'][h][2:-1])==stock_code and int(_list['zxsz']['="合约数量"'][h])==vol and int(_list['zxsz']['="到期日期"'][h][2:-1])>int(date):
                            count=1
                            if _list['zxsz']['="展期标志"'][h][2:-1]=='提出展期-办理中':message.append(loan['stock'][i]+'zxsz展期成功\n     backup: e to '+str(loan['end'][i])+' \n     end: '+str(loan['end'][i])+' to 99999999\n');loan['backup'][i]=loan['end'][i];loan['end'][i]='99999999'
                            elif _list['zxsz']['="展期标志"'][h][2:-1]=='提出展期-待确认':continue
                            elif _list['zxsz']['="展期标志"'][h][2:-1]=='未展期' or _list['zxsz']['="展期标志"'][h][2:-1]=='已拒绝展期':message.append(loan['stock'][i]+'zxsz展期失败\n     backup: e to “ ”\n');loan['backup'][i]=np.nan
                            else:message1.append(str(loan['stock'][i])+'zxsz,审批状态异常\n')
                            break
            elif broker=='zx':
                if 'zxsh' in _list:
                    for m in range(len(_list['zxsh'])):
                        if _list['zxsh']['="证券代码"'][m][2:-1]==stock_code and int(_list['zxsh']['="合约数量"'][m])==vol and int(_list['zxsh']['="到期日期"'][m][2:-1])>int(date):
                            count=1
                            if _list['zxsh']['="展期标志"'][m][2:-1]=='提出展期-办理中':message.append(loan['stock'][i]+'zxsh展期成功\n     backup: e to '+str(loan['end'][i])+' \n     end: '+str(loan['end'][i])+' to 99999999\n');loan['backup'][i]=loan['end'][i];loan['end'][i]='99999999'
                            elif _list['zxsh']['="展期标志"'][m][2:-1]=='提出展期-待确认':continue
                            elif (_list['zxsh']['="展期标志"'][m][2:-1]=='未展期' or _list['zxsh']['="展期标志"'][m][2:-1]=='已拒绝展期'):message.append(loan['stock'][i]+'zxsh展期失败\n     backup: e to “ ”\n');loan['backup'][i]=np.nan
                            else:message1.append(str(loan['stock'][i])+'zxsh,审批状态异常\n')
                            break
            elif broker=='gjsz':
                if 'gjsz' in _list:
                    for n in range(len(_list['gjsz'])):
                        #print(_list['gjsz'])
                        if '0'*(6-len(str(_list['gjsz']['证券代码'][n])))+str(_list['gjsz']['证券代码'][n])==stock_code and int(_list['gjsz']['申请股数'][n])==vol and int(_list['gjsz']['原合约到期日'][n].replace('-',''))>int(date):
                            count=1
                            if _list['gjsz']['审批状态'][n] in ['审批通过','展期成功']:message.append(loan['stock'][i]+'gjsz展期成功\n     backup: e to '+str(loan['end'][i])+' \n     end: '+str(loan['end'][i])+' to 99999999\n');loan['backup'][i]=loan['end'][i];loan['end'][i]='99999999'
                            elif _list['gjsz']['审批状态'][n]=='审批拒绝' :message.append(loan['stock'][i]+'gjsz展期失败\n     backup: e to “ ”\n');loan['backup'][i]=np.nan
                            else:message1.append(str(loan['stock'][i])+'gjsz,审批状态异常\n')
                            break
                        last = []
                        '''
                        if '0'*(6-len(str(_list['gjsz']['证券代码'][n])))+str(_list['gjsz']['证券代码'][n])==stock_code and int(_list['gjsz']['申请股数'][n]) < vol and int(_list['gjsz']['原合约到期日'][n].replace('-',''))>int(date):
                            if last == []:
                               last.append(int(_list['gjsz']['申请股数'][n]));last.append(stock_code);last.append(int(_list['gjsz']['展期后到期日'][n].replace('-','')))
                            elif int(_list['gjsz']['申请股数'][n])+last[0] == vol and stock_code == last[1] and int(_list['gjsz']['展期后到期日'][n].replace('-','')) == last[2]:
                                message.append(loan['stock'][i]+'gjsz展期成功\n     backup: e to '+str(loan['end'][i])+' \n     end: '+str(loan['end'][i])+' to #99999999\n');loan['backup'][i]=loan['end'][i];loan['end'][i]='99999999'
                            else:
                               message1.append(str(loan['stock'][i])+'gjsz,审批状态异常\n')
                        '''
            if count==0:
                message1.append(str(loan['stock'][i])+str(loan['broker'][i]+'未找到展期信息\n'))
    mes=[]
    if len(message)!=0:
        mes=message[0]
        if len(message)>1:
            for t in range(len(message)-1):
                mes+=message[t+1]
    mes1=[]
    if len(message1)!=0:
        mes1=message1[0]
        if len(message1)!=1:
            for gg in range(len(message1)-1):
                mes1+=message1[gg+1]
        #loan.to_csv(os.path.join('/old_home/data/public/intern/zrsun/yoyaku',date,date+'change.csv'),index=False)
    print(mes[:-1] if mes else mes1)
    loan=left_loan.append(loan);loan=loan.append(right_loan)
    if mes:sendmsg(mes[:-1])
    if mes1:sendmsg(mes1[:-1])
    if len(sys.argv)>=3:
        quit()
    loan.to_csv('/home/largefile/ftao/yoyaku_loan.csv',index=False)