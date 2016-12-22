# -*- coding: utf-8 -*-
"""

@author: Aaron_Huang
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import  csv
import sys
import time
import matplotlib.pylab as plt
import pdb
from itertools import islice
from collections import defaultdict
import datetime


def csv_read(file,hasHead,symbol):
    """
    csv文件读取为list,其中csv文件有

    Args:
        file: 改文件为每天所有城市订单总数
        hasHead: 取值为0或1，0表示从第一行开始读，1表示从第二行开始读
        symbol: 每一行的分隔符
    Returns:
        result: 返回一个list，记录每个产品价钱
    """
    result = []
    f = open(file,'r')
    for line in islice(f,hasHead,None):
        line = line.strip().split(symbol)
        tmplist = []
        for i in range(len(line)):
            tmplist.append(line[i])
        result.append(tmplist)
    return result
        
def combined_with_historydata(historyfile,todayfile,combinedfile):
    """
    将历史数据与当前数据结合，用于隔天预测的输入文件
    Args:
        historyfile: 输入历史数据
        todayfile: 当前数据
        combinedfile： 历史与当前数据结合后的新数据
    """
    result = {}
    key = [i+68 for i in range(298)]
    for key in key:
        result.setdefault(key,[])
    f = pd.read_csv(historyfile)
    for i in range(len(f)):
        tmplist = []
        tmplist.append(f['date'][i])
        tmplist.append(f['orderscnt'][i])
        result[f['cityid'][i]].append(tmplist)
    today = csv_read(todayfile,1,'\t')
    for data in today:
        tmplist = []
        tmplist.append(data[1])
        tmplist.append(int(data[2]))
        result[int(data[0])].append(tmplist)
    #保存历史以及最新数据的文件
    csvFile = open(combinedfile,'w',newline="")
    writer = csv.writer(csvFile)
    writer.writerow(['date','orderscnt','cityid'])
    for city in result:
        for data in result[city]:
            data.append(city)
            writer.writerow(data)
    csvFile.close()

def signal_arima_model(ts,inp,ind,inq):
    """
    给定一个时间序列，输出相应的预测结果

    Args:
        ts: 输入的时间序列

    Returns:
        predict_result: 输出预测的结果
    """
    try:
        #用之前最好参数运行模型
        model = ARIMA(ts, order=(inp,ind,inq))
        bestmodel = model.fit(disp=-1)
        bestd = ind
    except Exception as e:
        #如果使用之前参数，模型不能收敛，则重新寻求最优模型参数
        print("rerun model again!")
        minAIC = sys.maxsize
        bestp,bestd,bestq = 0,0,0
        for p in range(6):
            for d in range(3):
                for q in range(6):
                    try:
                        model = ARIMA(ts, order=(p, d, q))          
                        results_ARIMA = model.fit(disp=-1)
                        if results_ARIMA.aic<minAIC:
                            minAIC = results_ARIMA.aic
                            bestp,bestd,bestq = p,d,q
                            bestmodel = results_ARIMA
                    except Exception as e:
                        pass
    else:
        pass
    finally:
        pass
    predicts = bestmodel.predict(start=len(ts),end=len(ts))
    if bestd == 0:
        predict_result = predicts
    elif bestd == 1:
        predict_result = predicts+ts[-1]
    else:
        predict_result = predicts+ts[-1]+(ts[-2]-ts[-3])
    return predict_result[0]

def test_signal_arima_model(ts):
    """
    给定一个时间序列，输出相应的预测结果

    Args:
        ts: 输入的时间序列

    Returns:
        predict_result: 输出预测的结果
    """
    #如果使用之前参数，模型不能收敛，则重新寻求最优模型参数
    print("This is test arima model")
    if len(ts) < 10:
        predict_result = ts[-1]
        bestp,bestd,bestq = -1,-1,-1
        return predict_result,bestp,bestd,bestq

    minAIC = sys.maxsize
    bestp,bestd,bestq = 0,0,0
    for p in range(6):
        for d in range(3):
            for q in range(6):
                try:
                    model = ARIMA(ts, order=(p, d, q))          
                    results_ARIMA = model.fit(disp=-1)
                    if results_ARIMA.aic<minAIC:
                        minAIC = results_ARIMA.aic
                        bestp,bestd,bestq = p,d,q
                        bestmodel = results_ARIMA
                except Exception as e:
                    pass
    isZero = True
    try:
        predicts = bestmodel.predict(start=len(ts),end=len(ts))
    except Exception as  e:
        predicts = bestmodel.predict(start=len(ts)-1,end=len(ts))
        predicts = predicts[len(ts)]
        isZero = False
    else:
        pass
    finally:
        pass
    if bestd == 0:
        predict_result = predicts
    elif bestd == 1:
        predict_result = predicts+ts[-1]
    else:
        predict_result = predicts+ts[-1]+(ts[-2]-ts[-3])
    if isZero:
        predict_result = predict_result[0]
    return predict_result,bestp,bestd,bestq


def arima_model(infile,outfile):
    """
    构建局部平滑的季节arima 模型用于农产品的预测

    Args:
        infile: 输入为历史产品价格的数据
        outfile: 输出文件为所有趋势有突变的农产品的未来预测值
    """
    csvFile = open(outfile,'w',newline="")
    pdq_File = open('bestpdq.csv','w',newline="")
    writer = csv.writer(csvFile)
    writer_pdq = csv.writer(pdq_File)
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')
    data = pd.read_csv(infile, parse_dates=['date'], index_col='date',date_parser=dateparse)
    # bestpdq = csv_read('bestpdq.csv',0,',')
    index = 0
    for city in data.groupby(data['cityid']):
        ts = city[1]['orderscnt']
        ts_log = np.log(ts)
        ts = np.exp(ts_log)
        # p = bestpdq[index][1]
        # d = bestpdq[index][2]
        # q = bestpdq[index][3]
        finalresult,bestp,bestd,bestq = test_signal_arima_model(ts)
        pdq = []
        pdq.append(city[0])
        pdq.append(bestp)
        pdq.append(bestd)
        pdq.append(bestq)
        predict_result = []
        predict_result.append(city[0])
        predict_result.append(finalresult)
        writer.writerow(predict_result)
        writer_pdq.writerow(pdq)
        index += 1
    csvFile.close()
    pdq_File.close()
    print(index)


if __name__ == '__main__':

    arima_model('./preprocess.csv',today+'_predict_result.csv')
    print("Finished!")

