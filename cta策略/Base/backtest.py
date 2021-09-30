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
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
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
        if len(self.data) == 0:
            print('数据长度不够')
            exit()
        self.cal_way = cal_way
        self.his_data = {}  # 历史数据
        self.last_signal = []  # 上期信号

        # 简单的账户信息，可以复杂化
        self.min_point = min_point  # 最小变动单位
        self.slip_point = slip_point  # 交易滑点，双边
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
            a = 0
            if num > 202:
                a = num - 200
            if self.cal_way == 'close':
                self.his_data['open'] = open_array[a:num+1]
                self.his_data['high'] = High_array[a:num+1]
                self.his_data['close'] = Close_array[a:num+1]
                self.his_data['low'] = Low_array[a:num+1]
                self.his_data['vol'] = Volume_array[a:num+1]
                self.his_data['last'] = Close_array[num]
                self.his_data['yes'] = Close_array[num-1]
                self.his_data['time'] = time
            if self.cal_way == 'open':
                self.his_data['open'] = open_array[a:num]
                self.his_data['high'] = High_array[a:num]
                self.his_data['close'] = Close_array[a:num]
                self.his_data['low'] = Low_array[a:num]
                self.his_data['vol'] = Volume_array[a:num]
                self.his_data['last'] = open_array[num]
                self.his_data['yes'] = open_array[num - 1]
                self.his_data['time'] = time
            self.signal_cal()  # 计算信号
        self.jz = pd.Series(self.capital_list, index=self.data.index)
        self.daily_jz = self.jz.resample('D', kind='period').last().ffill().dropna()
        if self.last_signal == []:
            return
        self.signal = pd.DataFrame(self.last_signal)
        self.signal.columns = ['signal', 'time']
        self.signal = self.signal.set_index('time')

        pass
    def signal_cal(self):
        signal = 1
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

    def target_position(self, hands, price):
        slip_price = self.min_point * self.slip_point
        # 同方向
        if np.sign(hands) == np.sign(self.last_hands):
            if hands > 0:  # 多头
                if abs(hands) > abs(self.last_hands):  # 加仓
                    pnl = self.last_hands * self.multip * (price - self.last_price)
                    # 计算加权成本
                    price = (self.last_hands * price + (price + slip_price) * (hands - self.last_hands)) / hands
                    pass

                if abs(hands) < abs(self.last_hands):  # 减仓
                    # 计算加权成本
                    price = (self.last_hands * price + (price - slip_price) * (hands - self.last_hands)) / hands
                    pnl = self.last_hands * self.multip * (price - self.last_price)
                    pass
                if abs(hands) == abs(self.last_hands):
                    pnl = self.last_hands * self.multip * (price - self.last_price)

            if hands < 0:  # 空头
                if abs(hands) > abs(self.last_hands):  # 加仓
                    pnl = self.last_hands * self.multip * (price - self.last_price)
                    # 计算加权成本
                    price = (abs(self.last_hands) * price + (price - slip_price) * (
                                abs(hands) - abs(self.last_hands))) / abs(hands)

                if abs(hands) < abs(self.last_hands):  # 减仓
                    # 计算加权成本
                    price = (abs(self.last_hands) * price + (price + slip_price) * (
                                abs(hands) - abs(self.last_hands))) / abs(hands)
                    pnl = self.last_hands * self.multip * (price - self.last_price)
                    pass
                if abs(hands) == abs(self.last_hands):
                    pnl = self.last_hands * self.multip * (price - self.last_price)

            if hands == 0:
                pnl = 0

        # 反向
        if np.sign(hands) != np.sign(self.last_hands):
            if self.last_hands < 0:  # 上期持有空头
                # 先平空头后开多头
                pnl = self.last_hands * self.multip * (price - self.last_price + slip_price)
                price += slip_price
                pass
            if self.last_hands > 0:  # 上期持有多头
                # 先平多头后开空头
                pnl = self.last_hands * self.multip * (price - self.last_price - slip_price)
                price -= slip_price
                pass
            if self.last_hands == 0:  # 上期不持仓
                pnl = 0
                if hands > 0:
                    price += slip_price
                if hands < 0:
                    price -= slip_price
            pass

        commission = abs(hands - self.last_hands) * price * self.multip * self.comission_rate

        self.pnl[self.num] = pnl
        self.commission[self.num] = commission
        self.capital = self.capital + pnl - commission
        self.last_hands = hands

        self.last_price = price

        self.capital_list[self.num] = self.capital

    def jz_plot(self, fig_name, filedir, is_show=True):
        base = self.data['Close'].resample('D', kind='period').last().ffill().dropna()
        df = pd.concat([base, self.daily_jz], axis=1).dropna()
        df = df / df.iloc[0, :]
        df.columns = ['base', 'stragety']

        ax = df.plot()
        if is_show:
            plt.show()
        fig = ax.get_figure()
        name = filedir + fig_name + '.png'
        if not os.path.exists('/'.join(filedir.split('/')[:-2])):
            os.mkdir('/'.join(filedir.split('/')[:-2]))
        if not os.path.exists(filedir):
            os.mkdir(filedir)
        fig.savefig(name)

    def analysis(self):
        ana = Analysis()
        self.result = ana.analysis(self.daily_jz)
        print(self.result)
        return self.result
    # def ana_self(self):

class ROC(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)

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

class Cmb(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, jz_dir):
        # 参数设置
        signal = pd.read_excel(jz_dir)
        signal = signal.set_index(signal.columns[0]).fillna(0).mean(axis=1)
        # signal = signal.set_index(signal.columns[0]).dropna().mean(axis=1)
        signal.index = pd.to_datetime(signal.index)
        self.signal = signal

    def signal_cal(self):
        time = self.his_data['time']
        # if time < min(self.signal.index):
        #     self.target_position(0, self.his_data['last'])
        #     return
        # if time > max(self.signal.index):
        #     self.target_position(0, self.his_data['last'])
        #     return
        if time not in self.signal.index:
            self.target_position(0, self.his_data['last'])
            return
        signal = self.signal.loc[time]
        if signal > 0:
            signal = 1
        if signal < 0:
            signal = -1
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        self.last_signal.append([signal, time])





