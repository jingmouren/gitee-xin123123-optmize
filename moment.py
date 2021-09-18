import pandas as pd
from Base import vector_backtest
from formula_package import moment_signal, ta
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
from scipy.special import comb, perm
import itertools
from tqdm import tqdm
from gplearn_fix.genetic import SymbolicTransformer, SymbolicRegressor
start_date = '20100101'
end_date = '20200601'

start_date = '20200601'
end_date = '20210629'

file_dir = r"./data/RB_1d.csv"
file_dir = r"./data/RB_1min.csv"
test1 = vector_backtest(start_date, end_date, file_dir, freq='30min',cal_way='open')

###### 参数优化 ##########
# result_list = pd.DataFrame()
# jz_list = pd.DataFrame()
# name_list = []
# for timeperiod in range(5, 240, 5):
#     for std in [0.5, 1, 1.5, 2, 2.5, 3]:
#
#         print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
#         signal = boll_signal(test1.data['Close'], timeperiod, std)
#         test1.add_stragety(signal=signal)
#         test1.run()
#         # test1.jz_plot()
#         result = test1.analysis()
#         result['timeperiod'] = timeperiod
#         result['std'] = std
#         if len(result_list) == 0:
#             result_list = result
#             jz_list = test1.jz
#         else:
#             result_list = pd.concat([result_list, result], axis=0)
#             jz_list = pd.concat([jz_list, test1.jz], axis=1)
#         name_list.append('timeperiod: ' + str(timeperiod) + '; std: ' + str(std))
# print(result_list)
# # result_list.to_csv('./参数表/bolling.csv')



###### 最优参数 ###########
range_d = 10

print('range_d: '+str(range_d))
signal = -moment_signal(test1.data['Close'], range_d)
test1.add_stragety(signal=signal.shift())
test1.run()
test1.jz_plot()
aa = test1.jz

