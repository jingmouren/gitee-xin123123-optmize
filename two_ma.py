import pandas as pd
from Base import vector_backtest
from formula_package import two_ma_signal, ta
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
from scipy.special import comb, perm
import itertools
from tqdm import tqdm
from gplearn_fix.genetic import SymbolicTransformer, SymbolicRegressor
start_date = '20100101'
end_date = '20200601'

file_dir = r"./data/RB_1d.csv"
test1 = vector_backtest(start_date, end_date, file_dir, cal_way='open')

###### 参数优化 ##########
# result_list = pd.DataFrame()
# jz_list = pd.DataFrame()
# name_list = []
# for short_p in range(5, 240, 5):
#     for long_p in range(10, 240, 10):
#         if short_p >= long_p or long_p/short_p >4:
#             continue
#         print('short_p: '+str(short_p)+'; long_p: '+str(long_p))
#         signal = two_ma_signal(test1.data['Close'], short_p, long_p)
#         test1.add_stragety(signal=signal)
#         test1.run()
#         # test1.jz_plot()
#         result = test1.analysis()
#         result['short_p'] = short_p
#         result['long_p'] = long_p
#         if len(result_list) == 0:
#             result_list = result
#             jz_list = test1.jz
#         else:
#             result_list = pd.concat([result_list, result], axis=0)
#             jz_list = pd.concat([jz_list, test1.jz], axis=1)
#         name_list.append('short_p: ' + str(short_p) + '; long_p: ' + str(long_p))
# print(result_list)
# result_list.to_csv('./参数表/two_ma.csv')



###### 最优参数 ###########
short_p = 2
long_p = 5

print('short_p: '+str(short_p)+'; long_p: '+str(long_p))
signal = two_ma_signal(test1.data['Close'], short_p, long_p)
test1.add_stragety(signal=signal)
test1.run()
test1.jz_plot()
aa = test1.jz

