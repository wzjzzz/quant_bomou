# '''
# 数据是 
# 246: /home/largefile/public/net_data/stockSelect/fmBase_%s.hdf的x和60min ytrue (Rtn[thisStock].lag[60min]-Rtn[000905.SH].lag[60min]),
# /home/largefile/public/HF_factors/f12/output_factors/lastday的k线因子

# 去除涨跌停/ST/上市时间小于20个交易日的股票后,
# 计算间隔10min的x和60min ytrue的rankic、14:56的x和EOD的股票rtn-zz500 rtn的rankic

# 500的eod数据在246: /home/largefile/public/eoddata/000905.SZ.csv
# 判断/home/largefile/public/net_data/stockSelect/fmBase_%s.hdf中某一列是否为x的逻辑是
# 以"[0]"结尾且不在TradingDay;LocalTime;UpdateTime;InstrumentId;midprice[thisStock][0];volume[thisStock][0];lastpx[000905.SH][0];isST;isNoTrade;Rtn[thisStock].lag[30min];Rtn[000905.SH].lag[30min];Rtn[thisStock].lag[60min];Rtn[000905.SH].lag[60min];Rtn[thisStock].lag[120min];Rtn[000905.SH].lag[120min];Rtn[thisStock].lag[240min];Rtn[000905.SH].lag[240min];index中
# '''

运行指令：nohup sh ./run.sh rank_ic.py > run_out.txt 2>&1 &

输出两个结果：
rank_ic_1min.csv   （间隔1min的x和60min ytrue的rank_ic） 间隔时间和多少分钟暂时还需在代码设置
rank_ic_eod.csv  	 （14:56的x和EOD的股票rtn-zz500 rtn的rankic）