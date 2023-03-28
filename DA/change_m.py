# -*- coding: utf-8 -*-
#从中金所获取换月信息

from bs4 import BeautifulSoup
import urllib.request
import time
import csv
import re
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr


today = time.strftime('%Y%m%d',time.localtime());print(today)	

def py_cffex():	
	count = 1
	while True:
		if count == 1:
			urlpage = 'http://www.cffex.com.cn/jystz/index.html'
		else:
			urlpage = 'http://www.cffex.com.cn/jystz/index_' + str(count) + '.html'
		page = urllib.request.urlopen(urlpage)
		soup = BeautifulSoup(page,'html.parser')
		table = soup.find('a',attrs={'title':'关于提示股指期货和股指期权合约交割相关事项的通知'})
		if table != None:
			new_urlpage = 'http://www.cffex.com.cn' + table['href']
			new_page = urllib.request.urlopen(new_urlpage)
			new_soup = BeautifulSoup(new_page,'html.parser')
			new_table = new_soup.find('div',attrs={'class':'jysggnr'})
			the_change_day = re.findall( '最后交易日为202[0-9]年1*[0-9]月[1-3]*[0-9]日' , new_table.text)
			if len(the_change_day) != 1:
				mail('中金所的通知有误','字符串匹配失败')
			month_last = the_change_day[0][6:].split('年')
			month_last[1] = month_last[1].split('月')
			year = month_last[0]
			month = month_last[1][0] if len(month_last[1][0]) == 2 else '0' + month_last[1][0]
			day = month_last[1][1][:-1] if len(month_last[1][1][:-1]) == 2 else '0' + month_last[1][1][:-1]
			change = year + month + day
			if month == today[4:6]:
				f = open('/home/largefile/public/change_month.txt', 'w')
				f.writelines(change)
				f.close()
				# mail('中金所通知！',change)
			break
		count += 1

def mail(subject,text):
        my_sender='970566946@qq.com'    # 发件人邮箱账号
        my_pass = 'dfdthfmkjygzbfff'           # 发件人邮箱密码(当时申请smtp给的口令)
        my_user='839333487@qq.com'      # 收件人邮箱账号
        try:
            msg=MIMEText(text,'HTML','utf-8')
            msg['From']=formataddr([my_sender,my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
            msg['To']=formataddr([my_user,my_user])              # 括号里的对应收件人邮箱昵称、收件人邮箱账号
            msg['Subject']= subject                # 邮件的主题，也可以说是标题
            #server=smtplib.SMTP("smtp.163.com", 25)  # 发件人邮箱中的SMTP服务器，端口是80
            server=smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是80
            server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
            server.sendmail(my_sender,my_user,msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
            server.quit()# 关闭连接
        except Exception:# 如果 try 中的语句没有执行
            print('发送失败\t\n')



#每月第一天将change_month.txt里日期改为None
with open('/home/largefile/public/change_month.txt', 'w+') as f:
    data = f.readlines()
    if today[-2:] == '01':
        f = open('/home/largefile/public/change_month.txt', 'w')
        f.writelines('None')
        f.close()
    elif not data or data[0] == 'None':
        py_cffex()