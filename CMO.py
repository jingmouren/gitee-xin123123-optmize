import pandas as pd
from formula_package import CMO_signal, ta
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
from Base import vector_backtest



start_date = '20100101'
end_date = '20200601'
file_dir = r"./data/RB_data.csv"
test1 = vector_backtest(start_date, end_date, file_dir, cal_way='open')

# 参数调优
# result_list = pd.DataFrame()
# jz_list = pd.DataFrame()
# name_list = []
# for cmo_length in range(5, 240, 5):
#     for m_length in range(5, 240, 5):
#         print('cmo_length: '+str(cmo_length)+'; m_length: '+str(m_length))
#         signal = CMO_signal(test1.data['Close'], cmo_length, m_length)
#         test1.add_stragety(signal=signal)
#         test1.run()
#         # test1.jz_plot()
#         result = test1.analysis()
#         result['cmo_length'] = cmo_length
#         result['m_length'] = m_length
#         if len(result_list) == 0:
#             result_list = result
#             jz_list = test1.jz
#         else:
#             result_list = pd.concat([result_list, result], axis=0)
#             jz_list = pd.concat([jz_list, test1.jz], axis=1)
#         name_list.append('cmo_length: '+str(cmo_length)+'; m_length: '+str(m_length))
# print(result_list)
# result_list.to_csv('./stragety_data/cmo.csv')
# jz_list.columns = name_list
# jz_list.to_csv('cmo_jz.csv')




# 最优参数
cmo_length = 55
m_length = 45
print('cmo_length: '+str(cmo_length)+'; m_length: '+str(m_length))
signal = CMO_signal(test1.data['Close'], cmo_length, m_length)
test1.add_stragety(signal=signal)
test1.run()
test1.jz_plot()
result = test1.analysis()




