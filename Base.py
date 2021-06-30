import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis


class vector_backtest:
    def __init__(self, start_date, end_date, file_dir, cal_way='open'):
        data = pd.read_csv(file_dir)
        data = data.set_index(data.columns[0])
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        self.data = data.loc[start_date:end_date, :]
        self.cal_way = cal_way
        pass
    def add_stragety(self, signal):
        self.signal = signal
        pass
    def run(self):
        data = self.data
        if self.cal_way == 'open':
            ret = (data['开盘价'].diff() / data['开盘价'].shift()).fillna(0)
            singal = (self.signal.shift(2)).fillna(0)  # 收盘价
        elif self.cal_way == 'close':
            ret = (data['收盘价'].diff() / data['收盘价'].shift()).fillna(0)
            singal = (self.signal.shift(1)).fillna(0)  # 收盘价
        self.baseret = ret
        self.basejz = (1 + self.baseret).cumprod()
        self.ret = ret * singal
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

