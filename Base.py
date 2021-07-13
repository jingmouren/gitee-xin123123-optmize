import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import re

class vector_backtest:
    def __init__(self, start_date, end_date, file_dir, freq='1d', cal_way='open'):
        data = pd.read_csv(file_dir)
        data = data.set_index(data.columns[0])
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        self.data = data.loc[start_date:end_date, :]
        self.cal_way = cal_way
        freq_name = re.sub('\d+', '', freq)
        num = re.sub('[a-z A-Z]+', '', freq)
        if freq_name == 'min':
            num = int(num)
            self.data = resample(self.data, num)
        pass

    def add_stragety(self, signal):
        self.signal = signal
        pass

    def run(self):
        data = self.data
        if self.cal_way == 'open':
            ret = (data['Open'].diff() / data['Open'].shift()).fillna(0)
            singal = (self.signal.shift(2)).fillna(0)  # 收盘价
        elif self.cal_way == 'close':
            ret = (data['Close'].diff() / data['Close'].shift()).fillna(0)
            singal = (self.signal.shift(1)).fillna(0)  # 收盘价
        self.baseret = ret
        self.basejz = (1 + self.baseret).cumprod()
        self.ret = ret * singal - np.abs(ret * singal)*0.003  # 千3手续费
        self.jz = (1 + self.ret).cumprod()
        self.singal = singal

    def jz_plot(self):
        df = pd.concat([self.basejz, self.jz], axis=1)
        df.columns = ['base', 'stragety']
        df.plot()
        # self.jz.plot()
        plt.show()

    def analysis(self):
        ana = Analysis()
        self.result = ana.analysis(self.jz)
        return self.result
        print(self.result)

def resample(fix_df, n):
    '''k线合成'''
    Open = fix_df['Open'].shift(n-1)
    Close = fix_df['Close']
    High = fix_df['High'].rolling(n).max()
    Low = fix_df['Low'].rolling(n).min()
    Volume = fix_df['Volume'].rolling(n).sum()
    df = pd.concat([Open, High, Low, Close, Volume], axis=1).dropna()
    return df.iloc[::n, :]