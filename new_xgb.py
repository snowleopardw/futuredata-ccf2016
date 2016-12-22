# -*- coding: utf-8 -*-
"""
@author: Aaron Huang
"""
import os
mingw_path = 'D:/after-install/MinGW64/mingw64/bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import os
import time
from datetime import datetime, timedelta
from sklearn import preprocessing
import pdb
import matplotlib.pyplot as plt
import pdb
import csv
import stats as sts 
import traceback
import xgboost as xgb
    
def train_parameter(dataset,look_back,data_market,tmptype,data_name):
    """
    xgb模型参数训练

    Args:
        dataset: 总数据
        look_back: 滑动窗口大小
        data_market：市场名称映射值
        tmptype：产品类型
        data_name：农产品名称映射值
    Returns:
        返回最好的xgb模型参数
    """
    eta = 0.02
    max_depth = 6
    subsample = 0.85
    colsample_bytree = 0.85
    # print('XGboost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eval_metric": "rmse",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "seed": 1994,
    }
    num_boost_round = 10000
    early_stopping_rounds = 100
    
    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
    # print(len(train), len(test))
    try:
        # reshape into X=t and Y=t+1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        dtrain = xgb.DMatrix(trainX, trainY)
        dvalid = xgb.DMatrix(testX, testY)
        
        watchlist = [(dtrain, 'train'), (dvalid, 'test')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                        early_stopping_rounds=early_stopping_rounds)
        #保存最好的模型参数
        parameter_dict[data_market+'_'+tmptype+'_'+data_name] = gbm.best_iteration
    except Exception as e:
        print(data_market,data_name,data_type)
         

def retrain_and_predict_all(dataset,look_back,num_boost_round,resultdate,data_market,tmptype,data_name):
    eta = 0.02
    max_depth = 6
    subsample = 0.85
    colsample_bytree = 0.85
    
    # print('XGboost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "reg:linear",
        "booster": "gbtree",
        "eval_metric": "rmse",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "seed": 2016,
    }
    num_boost_round = num_boost_round
        
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(dataset, look_back)

    dtrain = xgb.DMatrix(trainX, trainY)
    #train all data
    gbm = xgb.train(params, dtrain, num_boost_round,verbose_eval=False)
    predict_series = [dataset[-look_back:]]
    for date in resultdate:
        tmpprice = gbm.predict(xgb.DMatrix(predict_series), ntree_limit=gbm.best_iteration+1)
        
        tmpresult = [data_market,tmptype,data_name,date,tmpprice[0]]
        finalresult.append(tmpresult)
        predict_series = predict_series[0].tolist()
        del predict_series[0]
        predict_series.append(tmpprice[0])
        predict_series = [np.array(predict_series)]

def create_dataset(dataset, look_back=1):
    """
    构造数据样本

    Args:
        dataset: 总数据
        look_back: 滑动窗口大小，默认滑动窗口为1
    Returns:
        返回获取的样本数据
    """
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1):
        aaaa = dataset[i:(i+look_back)]
        aaaa = np.array(extend_feature(aaaa.tolist()))
        dataX.append(aaaa)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

def extend_feature(scores):
    """
    特征构造

    Args:
        scores: 原始滑动窗口获得的特征
    Returns:
        返回基于滑动窗口特征增加的统计特征
    """
    features = scores
    features.append(np.sum(scores)) #总数
    features.append(np.mean(scores)) #平均数
    features.append(np.median(scores)) #中位数
    # features.append(sts.mode(scores)) #众数
    features.append(sts.quantile(scores,p=0.25)) #上四分位
    features.append(sts.quantile(scores,p=0.75)) #上七分位
    features.append(np.max(scores)) #最大值
    features.append(np.min(scores)) #最小值
    features.append(np.max(scores)-np.min(scores)) #极差
    features.append(sts.quantile(scores,p=0.75)-sts.quantile(scores,p=0.25)) #四分位差
    features.append(np.var(scores)) #方差
    features.append(np.std(scores)/np.mean(scores)) #离散系数
    features.append(sts.skewness(scores)) #偏度
    features.append(sts.kurtosis(scores)) #峰度
    return features

if __name__ == '__main__':    

    train = pd.read_csv('./farming.csv')
    test = pd.read_csv('./product_market.csv')
    valid_357 = pd.read_csv('./3_5_7_min_withhead.csv')
    usemodel = pd.read_csv('./usemodel.csv')
    addData = pd.read_csv('./halfnochange_halfadd.csv')
    disData = pd.read_csv('./halfnochange_halfdiscount.csv')
    mappingMarket_test = set(test['市场名称映射值'])
    mappingName_test = set(test['农产品名称映射值'])
    train_selectby_Market_test = train[train['市场名称映射值'].isin(mappingMarket_test)]
    train_selectby_Name_test = train_selectby_Market_test[train_selectby_Market_test['农产品名称映射值'].isin(mappingName_test)]
    train_selectby_test = train_selectby_Name_test
    train_selectby_test_backup = train_selectby_test #选取的测试集的备份,同时过滤掉颜色，规格，单位，区域，入库
    train_selectby_test.drop('区域', axis=1, inplace=True)
    train_selectby_test.drop('规格', axis=1, inplace=True)
    train_selectby_test.drop('颜色', axis=1, inplace=True)
    train_selectby_test.drop('单位', axis=1, inplace=True)
    train_selectby_test.drop('数据入库时间', axis=1, inplace=True)
    filter_train_selectby_test = train_selectby_test
    filter_train_selectby_test_leasttwomonth = filter_train_selectby_test
    newdata_orderbyMNT = filter_train_selectby_test_leasttwomonth.sort_index(by=['市场名称映射值','农产品名称映射值','数据发布时间'])
    valid_357_sorted = valid_357.sort_index(by=['name','market'])

    pre_name,pre_type,pre_market = None,None,None

    resultdate = ['2016-07-01','2016-07-02','2016-07-03','2016-07-04','2016-07-05','2016-07-06','2016-07-07','2016-07-08','2016-07-09','2016-07-10',
             '2016-07-11','2016-07-12','2016-07-13','2016-07-14','2016-07-15','2016-07-16','2016-07-17','2016-07-18','2016-07-19','2016-07-20',
             '2016-07-21','2016-07-22','2016-07-23','2016-07-24','2016-07-25','2016-07-26','2016-07-27','2016-07-28','2016-07-29',
             '2016-07-30','2016-07-31']
    #train for parameters
    minCount = 30 #所选数据的最小的长度
    xgb_count = 0
    train_time_start = time.time()
    parameter_dict = {} 
    best_rmse = []
    best_iteration = []
    for data_357 in valid_357_sorted.values:
        data_market = data_357[0]
        data_type = data_357[1]
        data_name = data_357[2]
        if pre_name == data_name and pre_market == data_market and pre_type==data_type:
            continue
        else:
            isModel = False
            for modeldata in usemodel.values:
                if data_market == modeldata[0].split('_')[0] and data_name == modeldata[0].split('_')[1]:
                    isModel = True
            if isModel:
                tmpselect1 = newdata_orderbyMNT[newdata_orderbyMNT['农产品名称映射值']==data_name]
                tmpselect2 = tmpselect1[tmpselect1['农产品类别']==data_type]
                finalselect = tmpselect2[tmpselect2['市场名称映射值']==data_market]
                finalselect = finalselect[finalselect['数据发布时间']>'2016-03-10']
                # finalselect = finalselect[-minCount:]
                xgb_count += 1
                tmptype = finalselect[-1:]['农产品类别'].values[0]
                dataset = finalselect['平均交易价格'].values
                dataset.astype('float32')
                train_parameter(dataset,7,data_market,tmptype,data_name)
            pre_name = data_name
            pre_type = data_type
            pre_market = data_market
    print(xgb_count)
    parameter_name = 'testtesttest_3_10' #参数保 存文件名
    # 保存xgb参数
    with open(parameter_name+'.csv', 'w',newline='') as f:
        w = csv.DictWriter(f, parameter_dict.keys())
        w.writeheader()
        w.writerow(parameter_dict)

    # #保存价格没有变化的数据
    # nochangeList = []
    # pre_name,pre_type,pre_market = None,None,None
    # for data_357 in valid_357_sorted.values:
    #     # pdb.set_trace() 
    #     data_market = data_357[0]
    #     data_type = data_357[1]
    #     data_name = data_357[2]
    #     if pre_name == data_name and pre_market == data_market and pre_type==data_type:
    #         continue
    #     else:
    #         tmpselect1 = newdata_orderbyMNT[newdata_orderbyMNT['农产品名称映射值']==data_name]
    #         tmpselect2 = tmpselect1[tmpselect1['农产品类别']==data_type]
    #         finalselect = tmpselect2[tmpselect2['市场名称映射值']==data_market]
    #         # finalselect = finalselect[finalselect['数据发布时间']>'2016-04-31']
    #         finalselect = finalselect[-60:]
    #         length = len(finalselect)
    #         if finalselect['平均交易价格'].values[int(length/3)] == finalselect['平均交易价格'].values[int(length*3/4)] and finalselect['平均交易价格'].values[0] == finalselect['平均交易价格'].values[-1]:
    #             nochangeList.append([data_market,data_name,data_type])
    #         pre_name = data_name
    #         pre_market = data_market
    #         pre_type = data_type

    # # 读取xgb参数
    # parameter_name = '126_parameters_6_09_09_002_40_features'
    # parametersDict = {}
    # rowcount = 0 
    # with open(parameter_name+'.csv', mode='r') as infile:
    #     reader = csv.reader(infile)
    #     for rows in reader:
    #         if rowcount == 0:
    #             rowkey = rows
    #         else:
    #             rowvalue = rows
    #         rowcount += 1
    #     for index in range(len(rowvalue)):
    #         parametersDict[rowkey[index]] = rowvalue[index]

    # # retrain all data and predict
    # pre_name,pre_type,pre_market = None,None,None
    # finalresult = []
    # minCount= 40

    # count = 0
    # for data_357 in valid_357_sorted.values:
    #     data_market = data_357[0]
    #     data_type = data_357[1]
    #     data_name = data_357[2]
    #     if pre_name == data_name and pre_market == data_market and pre_type==data_type:
    #         continue
    #     else:
    #         count += 1
    #         tmpselect1 = newdata_orderbyMNT[newdata_orderbyMNT['农产品名称映射值']==data_name]
    #         tmpselect2 = tmpselect1[tmpselect1['农产品类别']==data_type]
    #         finalselect = tmpselect2[tmpselect2['市场名称映射值']==data_market]
    #         finalselect = finalselect[finalselect['数据发布时间']>'2016-04-31']
    #         # finalselect = finalselect[-minCount:]
    #         # pricelist = [value for value in finalselect[-3:]['平均交易价格'].values]
    #         price = tmpselect2[tmpselect2['市场名称映射值']==data_market][-1:]['平均交易价格'].values[0]
    #         tmptype = finalselect['农产品类别'].values[0]
    #         if [data_market,data_name,data_type] in nochangeList:
    #             # 最近60天都没有变化，直接用最后一天的值预测未来
    #             for date in resultdate:
    #                 tmpresult = [data_market,tmptype,data_name,date,price]
    #                 finalresult.append(tmpresult)
    #         else:
    #             if len(finalselect)>=minCount:
    #                 try:
    #                     iterator_num = int(parametersDict[data_market+'_'+tmptype+'_'+data_name])
    #                     if iterator_num == 0:
    #                         for date in resultdate:
    #                             tmpresult = [data_market,tmptype,data_name,date,price]
    #                             finalresult.append(tmpresult)
    #                     else:
    #                         # 用所有数据重新训练
    #                         dataset = finalselect['平均交易价格'].values
    #                         dataset.astype('float32')
    #                         retrain_and_predict_all(dataset,7,iterator_num,resultdate,data_market,tmptype,data_name)
    #                 except Exception as e:
    #                     for date in resultdate:
    #                         tmpresult = [data_market,tmptype,data_name,date,price]
    #                         finalresult.append(tmpresult)
    #             else:
    #                 # 不在xgb预测范围内的，直接用最后一天的值预测未来
    #                 for date in resultdate:
    #                     tmpresult = [data_market,tmptype,data_name,date,price]
    #                     finalresult.append(tmpresult)
    #         pre_name = data_name
    #         pre_type = data_type
    #         pre_market = data_market
    #         if count%50==0:
    #             print(count)
