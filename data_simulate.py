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
        lag['开盘价'] = lag['收盘价']
        lag['最高价'] = lag['收盘价']
        lag['最低价'] = lag['收盘价']
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
                tt['开盘价'] = tt['收盘价']
                tt['最高价'] = tt['收盘价']
                tt['最低价'] = tt['收盘价']
                temp_data = pd.DataFrame(tt + self.diff_.iloc[rnd[i], :]).T
                sim_data = pd.concat([sim_data, temp_data], axis=0)
        sim_data.index = self.data.index
        return sim_data

start_date = '20100101'
end_date = '20200601'
file_dir = r"./data/RB_data.csv"
# r_list = pd.DataFrame()
# for i in range(10):
#     sum1 = DataSim(start_date, end_date, file_dir)
#     sum1.relative_cal()
#     sim_data = sum1.random_cal()
#     sim_data.to_csv('./data/sim_RB.csv')
#     # a = pd.DataFrame(sim_data['收盘价'])
#     # a.columns = ['a']
#     # a.plot()
#     # plt.show()
#
#
#     test1 = vector_backtest(start_date, end_date, './data/sim_RB.csv', cal_way='open')
#     # ###### 最优参数 ###########
#     # timeperiod = 25
#     # std = 1
#     # print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
#     # signal = boll_signal(test1.data['收盘价'], timeperiod, std)
#     # test1.add_stragety(signal=signal)
#     # test1.run()
#     # test1.jz_plot()
#     # result = test1.analysis()
#     # if len(r_list) == 0:
#     #     r_list = result
#     # else:
#     #     r_list = pd.concat([r_list, result], axis=0)
#     # print(r_list)
#
#     result_list = pd.DataFrame()
#     jz_list = pd.DataFrame()
#     name_list = []
#     for timeperiod in range(5, 120, 5):
#         for std in [0.5, 1, 1.5, 2, 2.5, 3]:
#             print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
#             signal = boll_signal(test1.data['收盘价'], timeperiod, std)
#             test1.add_stragety(signal=signal)
#             test1.run()
#             # test1.jz_plot()
#             result = test1.analysis()
#             result['timeperiod'] = timeperiod
#             result['std'] = std
#             if len(result_list) == 0:
#                 result_list = result
#                 jz_list = test1.jz
#             else:
#                 result_list = pd.concat([result_list, result], axis=0)
#                 jz_list = pd.concat([jz_list, test1.jz], axis=1)
#             name_list.append('timeperiod: ' + str(timeperiod) + '; std: ' + str(std))
#     print(result_list)
#     result_list.to_csv(str(i)+'_result_all.csv')
#
# r_list.to_csv(str(i)+'_result_all.csv')


sum1 = DataSim(start_date, end_date, file_dir)
sum1.relative_cal()
sim_data = sum1.random_cal()
sim_data.to_csv('./data/sim_RB.csv')
a = pd.DataFrame(sim_data['收盘价'])
a.columns = ['a']
a.plot()
plt.show()


test1 = vector_backtest(start_date, end_date, './data/sim_RB.csv', cal_way='open')
# ###### 最优参数 ###########
timeperiod = 110
std = 0.5
print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
signal = boll_signal(test1.data['收盘价'], timeperiod, std)
test1.add_stragety(signal=signal)
test1.run()
test1.jz_plot()
result = test1.analysis()
