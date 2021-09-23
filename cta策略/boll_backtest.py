import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re
from cta策略.backtest import SimpleBacktest
import talib as ta

class BOLL(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.n, self.p = n
    def signal_cal(self):
        n = self.n
        p = self.p

        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return
        upper, middle, lower = ta.BBANDS(
            self.his_data['close'],
            timeperiod=n,
            # number of non-biased standard deviations from the mean
            nbdevup=p,
            nbdevdn=p,
            # Moving average type: simple moving average here
            matype=0)
        close = self.his_data['close'][-1]
        signal = 0
        if close > upper[-1]:
            signal = 1
        if close < lower[-1]:
            signal = -1
        if signal == 0:
            self.target_position(self.last_hands, self.his_data['last'])
            return
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass



n_list = [[10,20,60,120],[0.5,1,1.5,2]]  # 参数

stragety_name = 'BOLL_1d'  # 策略名
filedir = './result/螺纹/'  # 图片保存地址
pic_name = 'rb_' + stragety_name + "_参数："  # 图片名称
result_df = pd.DataFrame()
jz_df = pd.DataFrame()
name_list = []
for n1 in n_list[0]:
    for n2 in n_list[1]:
        n = [n1, n2]
        roc = BOLL(start_date='2013-01-01',
                  end_date='2018-01-01',
                  # end_date='2021-09-01',
                  comission_rate=0.001,
                  init_cash=10000000,
                  main_file='./行情数据库/螺纹/',
                  symbol='RB',
                  multip=10,  # 交易乘数
                  freq='1d',
                  cal_way='open')

        fig_name = pic_name + str(n)
        name_list.append(str(n))
        roc.param_add(n)
        roc.run()
        roc.jz_plot(fig_name, filedir+stragety_name+'/')
        result = roc.analysis()
        result['参数'] = str(n)
        if len(result_df) == 0:
            result_df = result
            jz_df = roc.jz
        else:
            result_df = pd.concat([result_df, result], axis=0, sort=True)
            jz_df = pd.concat([jz_df, roc.jz], axis=1, sort=True)
result_df = result_df.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
result_df.to_excel(filedir+stragety_name+'/绩效.xlsx')
jz_df.columns = name_list
jz_df.to_excel(filedir+stragety_name+'/净值.xlsx')
print(result_df)




