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

# result_list = pd.DataFrame()
# for cmo_length in range(5, 240, 5):
#     for m_length in range(5, 240, 5):
#         print('cmo_length: '+str(cmo_length)+'; m_length: '+str(m_length))
#         signal = CMO_signal(test1.data['收盘价'], cmo_length, m_length)
#         test1.add_stragety(signal=signal)
#         test1.run()
#         # test1.jz_plot()
#         result = test1.analysis()
#         result['cmo_length'] = cmo_length
#         result['m_length'] = m_length
#         if len(result_list) == 0:
#             result_list = result
#         else:
#             result_list = pd.concat([result_list, result], axis=0)
# print(result_list)
# result_list.to_csv('result.csv')

cmo_length = 55
m_length = 45
print('cmo_length: '+str(cmo_length)+'; m_length: '+str(m_length))
signal = CMO_signal(test1.data['收盘价'], cmo_length, m_length)
test1.add_stragety(signal=signal)
test1.run()
test1.jz_plot()
result = test1.analysis()
