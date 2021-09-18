import pandas as pd
from Base import vector_backtest
from formula_package import DMA_signal, ta
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
# for N2 in range(5, 120, 5):
#     for N1 in range(10, 240, 10):
#         for M in range(5, 30, 5):
#             if N1 <= N2 or N1/N2 > 4:
#                 continue
#             print('N1: '+str(N1)+'; N2: '+str(N2)+'; M: '+str(M))
#             signal = DMA_signal(test1.data['Close'], N1, N2, M)
#             test1.add_stragety(signal=signal)
#             test1.run()
#             # test1.jz_plot()
#             result = test1.analysis()
#             result['N1'] = N1
#             result['N2'] = N2
#             result['M'] = M
#             if len(result_list) == 0:
#                 result_list = result
#                 jz_list = test1.jz
#             else:
#                 result_list = pd.concat([result_list, result], axis=0)
#                 jz_list = pd.concat([jz_list, test1.jz], axis=1)
#             name_list.append('N1: '+str(N1)+'; N2: '+str(N2)+'; M: '+str(M))
# print(result_list)
# result_list.to_csv('./参数表/dma.csv')



###### 最优参数 ###########
# N1 = 10
# N2 = 5
# M = 5
#
# print('N1: '+str(N1)+'; N2: '+str(N2)+'; M: '+str(M))
# signal = DMA_signal(test1.data['Close'], N1, N2, M)
# test1.add_stragety(signal=signal)
# test1.run()
# test1.jz_plot()
# aa = test1.jz

