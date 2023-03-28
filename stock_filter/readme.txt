# 是否ST 在 246: /home/largefile/public/ashareIsST_1801_2209.csv
# 生成每个股票是否涨停的数据 
# 去除涨跌停/ST/上市时间小于等于20个交易日
# 前20交易日 日成交金额均值低于2千万
运行指令：nohup sh ./run.sh $1 $2 ￥3 > run_out.txt 2>&1 &
$1：运行code的文件名
$2：开始日期
$3：结束日期
其中金额和上市和已有的合并

isfilter_limit.py：
	涨跌停数据

isfilter_no_limit.py
	amt和list


本文件生成一个上面后三条，方便复用，结果在results里

amt_less_2.csv :  前20交易日 日成交金额均值低于2千万  （1表示低于）
list_less20.csv：上市时间小于等于20个交易日	（1表示小于）
limit_minbar_daily: 每天股票是否涨跌停		（1表示涨跌停）


结果校验：
1、涨跌停看了几个没问题
2、小于等于20日，现在是小于等于21日，代码108行改成>号即可
3、小于2kw 看了一只股600137.SH也没问题，前20天可能不准确，因为没有20天的数据