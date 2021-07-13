import pandas as pd
from Base import vector_backtest
from formula_package import Aberration_signal, ta
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
result_list = pd.DataFrame()
jz_list = pd.DataFrame()
name_list = []
for timeperiod in range(5, 240, 5):
    for std in [0.5, 1, 1.5, 2, 2.5, 3]:

        print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
        signal = Aberration_signal(test1.data['Close'], timeperiod, std)
        test1.add_stragety(signal=signal)
        test1.run()
        # test1.jz_plot()
        result = test1.analysis()
        result['timeperiod'] = timeperiod
        result['std'] = std
        if len(result_list) == 0:
            result_list = result
            jz_list = test1.jz
        else:
            result_list = pd.concat([result_list, result], axis=0)
            jz_list = pd.concat([jz_list, test1.jz], axis=1)
        name_list.append('timeperiod: ' + str(timeperiod) + '; std: ' + str(std))
print(result_list)
result_list.to_csv('./参数表/Aberration.csv')




###### 最优参数 ###########
# timeperiod = 20
# std = 0.5
#
# print('timeperiod: '+str(timeperiod)+'; std: '+str(std))
# signal = Aberration_signal(test1.data['Close'], timeperiod, std)
# test1.add_stragety(signal=signal)
# test1.run()
# test1.jz_plot()
# aa = test1.jz

