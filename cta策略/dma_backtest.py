import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re
from cta策略.backtest import SimpleBacktest


class DMA(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq, cal_way)
        self.dma = []
        self.ama = []
    def param_add(self, n):
        # 参数设置
        self.n1, self.n2 = n
    def signal_cal(self):
        n1 = self.n1
        n2 = self.n2

        if len(self.his_data['close']) < n2:
            self.target_position(0, self.his_data['last'])
            return
        sma = np.mean(self.his_data['close'][-n1:])
        lma = np.mean(self.his_data['close'][-n2:])
        dma = sma - lma

        self.dma.append(dma)
        if len(self.dma) < n1:
            self.target_position(0, self.his_data['last'])
            return
        ama = np.mean(self.dma[-n1:])
        self.ama.append(ama)
        if len(self.ama) < 2:
            self.target_position(0, self.his_data['last'])
            return
        if self.dma[-1] > 0 and self.dma[-1] - self.dma[-2] > 0 and self.ama[-1] > 0 and self.ama[-1] - self.ama[-2] > 0:
            signal = 1
        elif self.dma[-1] < 0 and self.dma[-1] - self.dma[-2] < 0 and self.ama[-1] < 0 and self.ama[-1] - self.ama[-2] < 0:
            signal = -1
        else:
            hands = self.last_hands
            self.target_position(hands, self.his_data['last'])
            return
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass



n_list = [[5,25],[10,50],[12,60],[14,70],[20,100]]  # 参数

stragety_name = 'DMA_1d'  # 策略名
filedir = './result/螺纹/'  # 图片保存地址
pic_name = 'rb_' + stragety_name + "_参数："  # 图片名称
result_df = pd.DataFrame()
jz_df = pd.DataFrame()
for n in n_list:
    roc = DMA(start_date='2013-01-01',
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




