import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re
from cta策略.backtest import SimpleBacktest


class ROC(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq, cal_way)

    def param_add(self, n):
        # 参数设置
        self.n = n
    def signal_cal(self):
        n = self.n
        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return
        signal = np.sign(self.his_data['close'][-1] / self.his_data['close'][-n] - 1)
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass



n_list = [3,5,9,10,12,14,20,60]  # 参数
stragety_name = 'ROC_1d'  # 策略名
filedir = './result/螺纹/'  # 图片保存地址
pic_name = 'rb_' + stragety_name + "_参数："  # 图片名称
result_df = pd.DataFrame()
jz_df = pd.DataFrame()
for n in n_list:
    roc = ROC(start_date='2013-01-01',
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
    result['参数'] = n
    if len(result_df) == 0:
        result_df = result
        jz_df = roc.jz
    else:
        result_df = pd.concat([result_df, result], axis=0, sort=True)
        jz_df = pd.concat([jz_df, roc.jz], axis=1, sort=True)
result_df = result_df.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
result_df.to_excel(filedir+stragety_name+'/绩效.xlsx')
jz_df.columns = n_list
jz_df.to_excel(filedir+stragety_name+'/净值.xlsx')
print(result_df)




