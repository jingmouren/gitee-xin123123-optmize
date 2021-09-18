import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
import matplotlib.pyplot as plt
from analysis_model import Analysis
import os
import re
import datetime
from cta策略.backtest import SimpleBacktest


class ROC(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, init_cash, main_file, symbol, multip, freq, cal_way)

    def param_add(self, pre_date, trade_date, jz_dir):
        # 参数设置
        self.pre_date = pre_date
        self.trade_date = trade_date
        base_jz = pd.read_excel(jz_dir)
        base_jz = base_jz.set_index(base_jz.columns[0])
        self.base_jz = base_jz
    def signal_cal(self):
        if (self.num - 1) % self.trade_date == 0:  # 每隔n个周期计算最优参数
            pass
            time = self.his_data['time']
            temp_df_all = self.base_jz.loc[:time, :].iloc[-self.pre_date:-1, :]
            test_n = int(len(temp_df_all.index) * 0.7)
            yz_n = len(temp_df_all.index) - test_n
            temp_df = temp_df_all

            ana = Analysis()
            result_df = pd.DataFrame()
            for n in range(len(temp_df.columns)):
                result = ana.analysis(temp_df.iloc[:, n])
                result2 = ana.analysis(temp_df.iloc[:test_n, n])['夏普比率'][0]
                result3 = ana.analysis(temp_df.iloc[-yz_n:, n])['夏普比率'][0]
                dif = (result3 - result2)
                result['diff_sharep'] = dif
                result['参数'] = n
                if len(result_df) == 0:
                    result_df = result
                else:
                    result_df = pd.concat([result_df, result], axis=0)


            self.best_param = int(result_df.sort_values(by='diff_sharep').iloc[-1, :]['参数'])
            # self.best_param = int(result_df.sort_values(by='夏普比率').iloc[-1, :]['参数'])
            self.best_sharpe = result_df[(result_df.参数).astype(int)==self.best_param]['夏普比率'][0]
            self.diff_sharep = result_df[(result_df.参数).astype(int)==self.best_param]['diff_sharep'][0]



        n = self.best_param
        print('n: '+str(n))
        print('self.best_sharpe: '+str(self.best_sharpe))
        print('self.diff_sharep: '+str(self.diff_sharep))

        if self.num < n or self.best_sharpe < 1:
            self.target_position(0, self.his_data['last'])
            return
        signal = np.sign(self.his_data['close'][-1] / self.his_data['close'][-n] - 1)
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass




pre_date = 360  # 参数寻优长度
trade_date = 60  # 回测长度

stragety_name = 'ROC_1d'  # 策略名
filedir = './result/螺纹/'  # 图片保存地址
pic_name = 'rb_' + stragety_name + "wfo"  # 图片名称
jz_dir = filedir + stragety_name + '/净值.xlsx'

start_date = '2013-01-01'
start_date = pd.to_datetime(start_date) + datetime.timedelta(pre_date+200)
roc = ROC(start_date=start_date,
          end_date='2018-01-01',
          comission_rate=0.001,
          init_cash=10000000,
          main_file='./行情数据库/螺纹/',
          symbol='RB',
          multip=10,  # 交易乘数
          freq='1d',
          cal_way='open')


roc.param_add(pre_date, trade_date, jz_dir)
roc.run()
roc.jz_plot(pic_name, filedir+stragety_name+'/')
result = roc.analysis()
result['参数'] = 'wfo'
jz_df = roc.jz

result = result.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
result.to_excel(filedir+stragety_name+'/wfo绩效.xlsx')
jz_df.to_excel(filedir+stragety_name+'/wfo净值.xlsx')
print(result)





