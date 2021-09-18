#!/usr/bin/env python
# coding=utf-8
# -*- coding: utf-8 -*-

import pandas as pd
from Base import vector_backtest
from formula_package2 import *
import matplotlib.pyplot as plt
from data_simulate import DataSim
import numpy as np
import math
from scipy.signal import savgol_filter
from sklearn.ensemble  import RandomForestClassifier     #导入需要的模块
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    start_date = '20100101'
    end_date = '20200601'

    file_dir = r"./data/RB_1d.csv"
    sum1 = DataSim(start_date, end_date, file_dir)
    sum1.relative_cal()
    sim_data = sum1.random_cal()
    sim_data.to_csv('./data/sim_RB.csv')
    a = pd.DataFrame(sim_data['Close'])

    #### 样本外模拟测试 #########

    file_dir = r"./data/sim_RB.csv"
    test1 = vector_backtest(start_date, end_date, file_dir, freq='1d', cal_way='open')

    timeperiod = 40
    std = 1

    print('timeperiod: ' + str(timeperiod) + '; std: ' + str(std))
    signal1 = boll_signal(test1.data['Close'], timeperiod, std)
    signal1 = pd.DataFrame(signal1).T

    cmo_length = 55
    m_length = 45
    print('cmo_length: ' + str(cmo_length) + '; m_length: ' + str(m_length))
    signal2 = CMO_signal(test1.data['Close'], cmo_length, m_length)
    signal2 = pd.DataFrame(signal2)

    N1 = 20
    N2 = 10
    M = 10
    print('N1: '+str(N1)+'; N2: '+str(N2)+'; M: '+str(M))
    signal3 = DMA_signal(test1.data['Close'], N1, N2, M)
    signal3 = pd.DataFrame(signal3)

    timeperiod = 20
    std = 0.5

    # print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
    # signal4 = boll_signal(test1.data['Close'], timeperiod, std)
    # signal4 = pd.DataFrame(signal4).T

    timeperiod = 115
    up = 0.8
    down = 0.2
    print('timeperiod: ' + str(timeperiod) + '; up: ' + str(up) + '; down: ' + str(down))
    signal5 = rsi_signal(test1.data['Close'], timeperiod, up, down)
    signal5 = pd.DataFrame(signal5)

    short_p = 10
    long_p = 20

    print('short_p: ' + str(short_p) + '; long_p: ' + str(long_p))
    signal6 = two_ma_signal(test1.data['Close'], short_p, long_p)
    signal6 = pd.DataFrame(signal6).T

    # timeperiod = 20
    # std = 0.5
    #
    # print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
    # signal7 = Aberration_signal(test1.data['Close'], timeperiod, std)
    timeperiod = 20
    signal7 = moment(test1.data['Close'], timeperiod)

    timeperiod = 60
    signal8 = moment(test1.data['Close'], timeperiod)
    timeperiod = 120
    signal9 = moment(test1.data['Close'], timeperiod)

    timeperiod = 20
    signal10 = vol(test1.data['Close'], timeperiod)
    timeperiod = 60
    signal11 = vol(test1.data['Close'], timeperiod)
    timeperiod = 120
    signal12 = vol(test1.data['Close'], timeperiod)

    X = pd.concat([signal1, signal2, signal3, signal5, signal6, signal7,signal8,signal9,signal10,signal1,signal12], axis=1)
    X = pd.concat([signal1, signal2, signal3, signal5, signal6],
                  axis=1)
    y = (sim_data['Open'].diff() / sim_data['Open'].shift()).fillna(0).shift(-2)
    y[y > 0] = 1
    y[y < 0] = -1
    df = pd.concat([y, X], axis=1).dropna()
    df = df.astype(np.float)
    score_list = pd.Series()
    score_listx = []
    for d in range(300, len(df.index)-100, 100):
        t_y = df.iloc[d-300:d, 0]
        t_X = df.iloc[d-300:d, 1:]
        test_y = df.iloc[d:d+100, 0]
        test_X = df.iloc[d:d+100, 1:]

        n_estimators = np.arange(100, 2000, step=100)
        max_features = ["auto", "sqrt", "log2"]
        max_depth = list(np.arange(2, 10, step=2)) + [None]
        min_samples_split = np.arange(2, 10, step=2)
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        param_grid = {"n_estimators": n_estimators, "max_features": max_features,
                      "max_depth": max_depth, "min_samples_split": min_samples_split,
                      "min_samples_leaf": min_samples_leaf, "bootstrap": bootstrap,}

        forest = RandomForestClassifier()
        random_cv = RandomizedSearchCV(forest, param_grid, n_iter=100, cv=5, n_jobs=-1)

        random_cv.fit(t_X, t_y)
        print("Best params:\n")
        print(random_cv.best_params_)
        print('score: ', random_cv.score(t_X, t_y))
        print('test_score: ', random_cv.score(test_X, test_y))

        # score_listx.append(random_cv.score(test_X, test_y))
        # print(score_listx)
        # y_pred = random_cv.predict(test_X)
        # y_pred = pd.Series(y_pred, index=test_X.index)
        # score_list = pd.concat([score_list, y_pred], axis=0)
        # # score_list.extend(y_pred)
        #
        # test1.add_stragety(signal=score_list)
        # test1.run()
        # test1.jz_plot()

        result = pd.DataFrame(random_cv.cv_results_).sort_values(by='mean_test_score').iloc[-5:, :]

        n_estimators = sorted(set(result['param_n_estimators']))
        max_features = sorted(set(result['param_max_features']))
        try:
            max_depth = sorted(set(result['param_max_depth']))
        except:
            max_depth = sorted(set(result['param_max_depth'].dropna())) + [None]
        min_samples_split = sorted(set(result['param_min_samples_split']))
        min_samples_leaf = sorted(set(result['param_min_samples_leaf']))
        bootstrap = sorted(set(result['param_bootstrap']))
        new_params = {"n_estimators": n_estimators, "max_features": max_features,
                      "max_depth": max_depth, "min_samples_split": min_samples_split,
                      "min_samples_leaf": min_samples_leaf, "bootstrap": bootstrap, }

        forest = RandomForestClassifier()
        grid_cv = GridSearchCV(forest, new_params, n_jobs=-1)

        grid_cv.fit(t_X, t_y)
        print('Best params:\n')
        print(grid_cv.best_params_, '\n')

        print('score: ', grid_cv.score(t_X, t_y))
        print('test_score: ', grid_cv.score(test_X, test_y))
        score_listx.append(grid_cv.score(test_X, test_y))
        print(score_listx)
        y_pred = grid_cv.predict(test_X)
        y_pred = pd.Series(y_pred, index=test_X.index)
        score_list = pd.concat([score_list, y_pred], axis=0)
        # score_list.extend(y_pred)

        test1.add_stragety(signal=score_list)
        test1.run()
        test1.jz_plot()

    test1.add_stragety(signal=score_list)
    test1.run()
    test1.jz_plot()

    aa = test1.jz