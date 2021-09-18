import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Base import vector_backtest
from formula_package import boll_signal, ta
import random

class DataSim:
    '''行情模拟'''
    def __init__(self, start_date, end_date, file_dir):
        data = pd.read_csv(file_dir)
        data = data.set_index(data.columns[0])
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        self.data = data.loc[start_date:end_date, :]
        pass
    def relative_cal(self):
        lag = self.data.shift(1)
        lag['Open'] = lag['Close']
        lag['High'] = lag['Close']
        lag['Low'] = lag['Close']
        self.diff_ = (self.data - lag).dropna()

    def random_cal(self, num=None):
        if num == None:
            num = len(self.data.index)
        rnd = np.random.randint(0, len(self.diff_.index), size=num)
        sim_data = pd.DataFrame()
        for i in range(0, num):
            if len(sim_data) == 0:
                sim_data = pd.DataFrame(self.data.iloc[0, :]).T
            else:
                tt = sim_data.iloc[-1, :]
                tt['Open'] = tt['Close']
                tt['High'] = tt['Close']
                tt['Low'] = tt['Close']
                temp_data = pd.DataFrame(tt + self.diff_.iloc[rnd[i], :]).T
                sim_data = pd.concat([sim_data, temp_data], axis=0)
        sim_data.index = self.data.index
        return sim_data

if __name__ == "__main__":
    start_date = '20100101'
    end_date = '20200601'
    file_dir = r"./data/RB_1d.csv"


    sum1 = DataSim(start_date, end_date, file_dir)
    sum1.relative_cal()
    sim_data = sum1.random_cal()
    sim_data.to_csv('./data/sim_RB.csv')
    a = pd.DataFrame(sim_data['Close'])
    a.columns = ['a']
    a.plot()
    plt.show()


    # test1 = vector_backtest(start_date, end_date, './data/sim_RB.csv', cal_way='open')
    # # ###### 最优参数 ###########
    # timeperiod = 110
    # std = 0.5
    # print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
    # signal = boll_signal(test1.data['收盘价'], timeperiod, std)
    # test1.add_stragety(signal=signal)
    # test1.run()
    # test1.jz_plot()
    # result = test1.analysis()
