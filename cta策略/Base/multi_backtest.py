import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return dirs
        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件

class MultiBacktest:
    def __init__(self, start_date, end_date, init_cash, symbol_list, freq='1d'):
        main_file = './result/'
        if symbol_list == 'all':
            symbol_list = file_name(main_file)
        jz_df = pd.DataFrame()
        signal_df = pd.DataFrame()
        for symbol in symbol_list:
            portfolio_jz = main_file + symbol + '/Portfolio_' + freq + '/wfo_组合净值.xlsx'
            portfolio_signal = main_file + symbol + '/Portfolio_' + freq + '/wfo_组合signal.xlsx'
            jz = pd.read_excel(portfolio_jz)
            jz = jz.set_index(jz.columns[0])
            signal = pd.read_excel(portfolio_signal)
            signal = signal.set_index(signal.columns[0])
            jz_df = pd.concat([jz_df, jz], axis=1)
            signal_df = pd.concat([signal_df, signal], axis=1)
        jz_df.columns = symbol_list
        signal_df.columns = symbol_list
        jz_df = jz_df.loc[start_date:end_date, :]
        signal_df = signal_df.loc[start_date:end_date, :]
        print(jz_df.head())
        print(signal_df.head())
        self.init_cash = init_cash
        self.base_ret = (jz_df.diff()/jz_df.shift()).fillna(0)
        self.base_jz = jz_df
        self.his_data = {}
        self.ret_list = []
        self.time_list = []
        pass
    def run(self):
        time_array = np.array(self.base_ret.index)
        ret = np.array(self.base_ret)
        base_jz = np.array(self.base_jz)
        for num, time in enumerate(time_array):
            self.num = num
            self.time = time
            if num == 0:
                continue
            a = 0
            if num > 202:
                a = num - 200
            self.his_data['ret'] = ret[num]
            self.his_data['base_jz'] = base_jz[a:num]

            self.signal_cal()  # 计算信号

        self.ret_list = pd.Series(self.ret_list, index=self.time_list)
        self.jz = (self.ret_list+1).cumprod() * self.init_cash
        self.daily_jz = self.jz.resample('D', kind='period').last().ffill().dropna()

        pass
    def signal_cal(self):
        jz = pd.DataFrame(self.his_data['base_jz'][-90:])
        # print(jz)
        result = jz.iloc[-1, :] / jz.iloc[0,:]
        weight = result/result.sum()
        self.position_cal(weight)
        return weight

    def position_cal(self, weight):
        ret = sum(self.his_data['ret'] * weight)
        self.ret_list.append(ret)
        self.time_list.append(self.time)

    def jz_plot(self, fig_name, filedir):
        base = (1 + self.base_ret.mean(axis=1)).cumprod() * self.init_cash
        base = base.resample('D', kind='period').last().ffill().dropna()
        df = pd.concat([base, self.daily_jz], axis=1).dropna()
        df = df / df.iloc[0, :]
        df.columns = ['base', 'stragety']
        df['extra'] = df['stragety'].diff()/df['stragety'].shift() - df['base'].diff()/df['base'].shift()
        df['extra'] = (1 + df['extra'].fillna(0)).cumprod()
        ax = df.plot()
        # df.plot()
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
