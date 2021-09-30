import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re
from cta策略.Base.multi_backtest import MultiBacktest

class MyMultiBacktest(MultiBacktest):
    def __init__(self, start_date, end_date, init_cash, symbol_list, freq='1d'):
        super().__init__(start_date, end_date, init_cash, symbol_list, freq)

    def signal_cal(self):
        jz = pd.DataFrame(self.his_data['base_jz'][-360:])
        if len(jz) < 30:
            return [1/len(jz.columns)] * len(jz.columns)
        # print(jz)
        result = jz.iloc[-1, :] / jz.iloc[0,:]
        result_list = []
        for i in range(len(jz.columns)):
            c = np.array(jz.iloc[:, i])
            if c[10] == c[0]:
                result = 0
            else:
                result = np.mean(c)/(np.std(c) + 0.00000000001)
                result = c[-1]/c[0]

            result_list.append(result)
        weight = result_list/np.sum(result_list)
        # weight = (max(result_list)+np.array(result_list))/sum(result_list)
        self.position_cal(weight)


start_date = '2014-01-01'
end_date = '2018-01-01'
init_cash = 10000000
freq='1d'
filedir = './portfolio/' + freq


pic_name = 'portfolio_'+freq  # 图片名称
cmb = MyMultiBacktest(start_date=start_date,
                    end_date=end_date,
                    init_cash=init_cash,
                    symbol_list='all',
                    freq=freq)
cmb.run()
cmb.jz_plot(pic_name, filedir+'/')

result = cmb.analysis()
result['参数'] = '组合'
result = result.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
final_jz = cmb.jz
result.to_excel(filedir+'/'+pic_name+'组合绩效.xlsx')
final_jz.to_excel(filedir+'/'+pic_name+'组合净值.xlsx')