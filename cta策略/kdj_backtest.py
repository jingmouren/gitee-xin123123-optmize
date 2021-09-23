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

class KDJ(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.short, self.long, self.mmid = n
    def signal_cal(self):
        n1 = self.short
        n2 = self.long
        n3 = self.mmid

        if len(self.his_data['close']) < n1 + n2+n3:
            self.target_position(0, self.his_data['last'])
            return
        slowk, slowd = ta.STOCH(self.his_data['high'],
                                self.his_data['low'],
                                self.his_data['close'],
                                fastk_period=n1,
                                slowk_period=n2,
                                slowk_matype=0,
                                slowd_period=n3,
                                slowd_matype=0)
        K = slowk[-1]
        D = slowd[-1]
        J = (3 * K) - (2 * D)
        signal = 0
        if K > D:
            signal = 1
        if K < D:
            signal = -1

        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass



n_list = [[4,2,2],[9,3,3],[16,4,4],[25,5,5],[36,6,6]]  # 参数

stragety_name = 'KDJ_1d'  # 策略名
filedir = './result/螺纹/'  # 图片保存地址
pic_name = 'rb_' + stragety_name + "_参数："  # 图片名称
result_df = pd.DataFrame()
jz_df = pd.DataFrame()
for n in n_list:
    roc = KDJ(start_date='2013-01-01',
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
jz_df.columns = [str(x) for x in n_list]
jz_df.to_excel(filedir+stragety_name+'/净值.xlsx')
print(result_df)




