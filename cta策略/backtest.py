import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re

# class Account:
#     def __init__(self, init_cash, comission_rate):
#         self.init_cash = init_cash
#         self.comission_rate = comission_rate
#         self.buy_position = {}  # 多头持仓
#         self.sell_position = {}  # 空头持仓
#         self.pnl = 0  # 单笔盈亏
#         self.cun_pnl = 0  # 累计盈亏
#         self.comission = 0  # 单笔手续费
#         self.cum_comission = 0  # 累计手续费
#     def target_position(self, order_dict):
#         for key, hands in order_dict.items():
#             if key not in self.buy_position:
#                 self.buy_position[key] = hands
#                 # self.comission +=
#             pass

class SimpleBacktest:
    def __init__(self, start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        if freq == '1d':
            file_dir = main_file + '日频/' + symbol + '_' + str(freq) + '.csv'
        else:
            file_dir = main_file + '分钟/' + symbol + '_' + str(freq) + '.csv'
        data = pd.read_csv(file_dir)
        data = data.set_index(data.columns[0])
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
        self.start_date = start_date
        self.end_date = end_date
        self.data = data.loc[start_date:end_date, :]
        self.cal_way = cal_way
        self.his_data = {}  # 历史数据
        self.last_signal = 0  # 上期信号

        # 简单的账户信息，可以复杂化
        self.multip = multip  # 交易乘数
        self.capital = init_cash
        self.capital_list = np.zeros(len(self.data.index))
        self.capital_list[0] = init_cash  # 动态权益
        self.last_hands = 0  # 上期持仓
        self.last_price = 0  # 上期持仓价格
        self.pnl = np.zeros(len(self.data.index))  # 收益
        self.comission_rate = comission_rate  # 手续费比率，双边收费
        self.commission = np.zeros(len(self.data.index))  # 手续费
        pass
    def run(self):
        time_array = np.array(self.data.index)
        open_array = np.array(self.data.Open)
        High_array = np.array(self.data.High)
        Close_array = np.array(self.data.Close)
        Low_array = np.array(self.data.Low)
        Volume_array = np.array(self.data.Volume)
        for num, time in enumerate(time_array):
            self.num = num
            if num == 0:
                continue
            if self.cal_way == 'close':
                self.his_data['open'] = open_array[:num+1]
                self.his_data['high'] = High_array[:num+1]
                self.his_data['close'] = Close_array[:num+1]
                self.his_data['low'] = Low_array[:num+1]
                self.his_data['vol'] = Volume_array[:num+1]
                self.his_data['last'] = Close_array[num]
                self.his_data['yes'] = Close_array[num-1]
                self.his_data['time'] = time
            if self.cal_way == 'open':
                self.his_data['open'] = open_array[:num]
                self.his_data['high'] = High_array[:num]
                self.his_data['close'] = Close_array[:num]
                self.his_data['low'] = Low_array[:num]
                self.his_data['vol'] = Volume_array[:num]
                self.his_data['last'] = open_array[num]
                self.his_data['yes'] = open_array[num - 1]
                self.his_data['time'] = time
            self.signal_cal()  # 计算信号
        self.jz = pd.Series(self.capital_list, index=self.data.index)
        pass
    def signal_cal(self):
        signal = 1
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

    def target_position(self, hands, price):
        pnl = self.last_hands * self.multip * (price - self.last_price)
        commission = abs(hands - self.last_hands) * price * self.multip * self.comission_rate
        self.pnl[self.num] = pnl
        self.commission[self.num] = commission
        self.capital = self.capital + pnl - commission
        self.last_hands = hands
        self.last_price = price

        self.capital_list[self.num] = self.capital

    def jz_plot(self, fig_name, filedir):
        df = pd.concat([self.data['Close'], self.jz], axis=1).dropna()
        df = df / df.iloc[0, :]
        df.columns = ['base', 'stragety']
        ax = df.plot()
        # df.plot()
        plt.show()
        fig = ax.get_figure()
        name = filedir + fig_name + '.png'
        if not os.path.exists(filedir):
            os.mkdir(filedir)
        fig.savefig(name)

    def analysis(self):
        ana = Analysis()
        self.result = ana.analysis(self.jz)
        print(self.result)
        return self.result
    # def ana_self(self):

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







