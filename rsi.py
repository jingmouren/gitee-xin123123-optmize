import pandas as pd
from Base import vector_backtest
from formula_package import rsi_signal, ta
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
from scipy.special import comb, perm
import itertools
from tqdm import tqdm
from gplearn_fix.genetic import SymbolicTransformer, SymbolicRegressor
start_date = '20100101'
end_date = '20200601'
file_dir = r"./data/RB_data.csv"
test1 = vector_backtest(start_date, end_date, file_dir, cal_way='open')

###### 参数优化 ##########
# result_list = pd.DataFrame()
# jz_list = pd.DataFrame()
# name_list = []
# for timeperiod in range(5, 120, 10):
#     for up in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
#         for down in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
#             print('timeperiod: '+str(timeperiod)+'; up: '+str(up)+'; down: '+str(down))
#             signal = rsi_signal(test1.data['Close'], timeperiod, up, down)
#             test1.add_stragety(signal=signal)
#             test1.run()
#             # test1.jz_plot()
#             result = test1.analysis()
#             result['timeperiod'] = timeperiod
#             result['up'] = up
#             result['down'] = down
#             if len(result_list) == 0:
#                 result_list = result
#                 jz_list = test1.jz
#             else:
#                 result_list = pd.concat([result_list, result], axis=0)
#                 jz_list = pd.concat([jz_list, test1.jz], axis=1)
#             name_list.append('timeperiod: '+str(timeperiod)+'; up: '+str(up)+'; down: '+str(down))
# print(result_list)
# result_list.to_csv('rsi.csv')
# jz_list.columns = name_list
# jz_list.to_csv('rsi_jz.csv')


###### 最优参数 ###########
timeperiod = 115
up = 0.95
down = 0.3
print('timeperiod: '+str(timeperiod)+'; up: '+str(up)+'; down: '+str(down))
signal = rsi_signal(test1.data['Close'], timeperiod, up, down)
test1.add_stragety(signal=signal)
test1.run()
test1.jz_plot()
aa = test1.jz


