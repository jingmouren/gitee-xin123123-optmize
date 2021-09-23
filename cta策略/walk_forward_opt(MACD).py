import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re
import datetime
from cta策略.backtest import SimpleBacktest
import talib as ta

class MACD(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq, cal_way)
    def param_add(self, pre_date, trade_date, jz_dir):
        # 参数设置
        self.pre_date = pre_date
        self.trade_date = trade_date
        base_jz = pd.read_excel(jz_dir)
        base_jz = base_jz.set_index(base_jz.columns[0])
        self.base_jz = base_jz
        if self.start_date > max(self.base_jz.index):
            print('单策略数据长度不够！！！')
            print('last history time: ', max(self.base_jz.index))
            print('start time: ', self.start_date)
            exit()

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def signal_cal(self):
        if (self.num - 1) % self.trade_date == 0:  # 每隔n个周期计算最优参数
            pass
            time = self.his_data['time']
            temp_df_all = self.base_jz.loc[:time, :].iloc[-self.pre_date:-1, :]
            test_n = int(len(temp_df_all.index) * 0.7)
            yz_n = len(temp_df_all.index) - test_n
            temp_df = temp_df_all

            ret = (temp_df.diff() / temp_df.shift()).dropna()
            # ret_mean = ret.mean(axis=1)
            # ret_mean[ret_mean<0] = 0
            # ret = ret.sub(ret_mean, axis=0)
            ret[ret > 0] = 1
            ret[ret < 0] = 0
            freq_ret = ret.mean()
            n = 2
            self.best_param = freq_ret.sort_values().index[-n:]
            self.best_freq = list(freq_ret.sort_values())[-n:]
            ana = Analysis()
            self.best_sharpe = []
            for n in self.best_param:
                result = ana.analysis(temp_df.loc[:, n])
                self.best_sharpe.append(result['夏普比率'][0])

        if self.num < max([np.float(x) for x in re.findall("\d+",str(list(self.best_param)))]) or np.mean(self.best_sharpe) < 0:
            self.target_position(0, self.his_data['last'])
            return
        signal_list = []
        for n in self.best_param:
            n = eval(n)
            short = n[0]
            long = n[1]
            mmid = n[2]
            if len(self.his_data['close']) < long + mmid:
                self.target_position(0, self.his_data['last'])
                return
            macd, signal, hist = ta.MACD(self.his_data['close'], fastperiod=short, slowperiod=long, signalperiod=mmid)
            signal = 0

            if macd[-1] > 0:
                signal = 1
            elif macd[-1] < 0:
                signal = -1

            signal_list.append(signal)
        signal = np.sign(np.sum(signal_list))
        # if signal == 0:
        #     self.target_position(self.last_hands, self.his_data['last'])
        #     return
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])





pre_date = 180  # 参数寻优长度
trade_date = 90  # 回测长度

stragety_name = 'MACD_1d'  # 策略名
filedir = './result/螺纹/'  # 图片保存地址
pic_name = 'rb_' + stragety_name + "wfo"  # 图片名称
jz_dir = filedir + stragety_name + '/净值.xlsx'

start_date = '2013-01-01'
# start_date = '2018-01-01'
start_date = pd.to_datetime(start_date) + datetime.timedelta(pre_date+200)
roc = MACD(start_date=start_date,
          end_date='2018-01-01',
          # end_date='2021-09-01',
          comission_rate=0.001,
          init_cash=10000000,
          main_file='./行情数据库/螺纹/',
          symbol='RB',
          multip=10,  # 交易乘数
          freq='1d',
          cal_way='open')


roc.param_add(pre_date, trade_date, jz_dir)
roc.run()
roc.jz_plot(pic_name, filedir+stragety_name+'/')
result = roc.analysis()
result['参数'] = 'wfo'
jz_df = roc.jz

result = result.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
result.to_excel(filedir+stragety_name+'/wfo绩效.xlsx')
jz_df.to_excel(filedir+stragety_name+'/wfo净值.xlsx')
print(result)





