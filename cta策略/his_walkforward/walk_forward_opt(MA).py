import pandas as pd
import numpy as np
from analysis_model import Analysis
import datetime
from cta策略.Base.backtest import SimpleBacktest, Cmb
import talib as ta

class MA(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, pre_date, trade_date, jz_dir, best_param_num=2):
        # 参数设置
        self.pre_date = pre_date
        self.trade_date = trade_date
        base_jz = pd.read_excel(jz_dir)
        base_jz = base_jz.set_index(base_jz.columns[0])
        self.base_jz = base_jz
        if self.start_date > max(self.base_jz.index):
            print('单策略数据长度不够！！！')
            print('last history time: ', max(self.base_jz.index))
            print('start time: ', self.start_date)
            exit()
        self.best_param_num = best_param_num

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def signal_cal(self):
        if (self.num - 1) % self.trade_date == 0:  # 每隔n个周期计算最优参数
            pass
            time = self.his_data['time']
            temp_df_all = self.base_jz.loc[:time, :].iloc[-self.pre_date:-1, :]
            # temp_df_all = self.base_jz.loc[:time, :].iloc[:-1, :]
            test_n = int(len(temp_df_all.index) * 0.7)
            yz_n = len(temp_df_all.index) - test_n
            temp_df = temp_df_all

            ret = (temp_df.diff() / temp_df.shift()).dropna()
            # ret_mean = ret.mean(axis=1)
            # ret_mean[ret_mean<0] = 0
            # ret = ret.sub(ret_mean, axis=0)
            freq_ret = ret.apply(lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if x[x < 0].sum() != 0 else 0)

            # ret[ret > 0] = 1
            # ret[ret < 0] = 0
            # freq_ret = ret.mean()
            n = self.best_param_num
            self.best_param = freq_ret.sort_values().index[-n]
            self.best_freq = list(freq_ret.sort_values())[-n]
            ana = Analysis()
            self.best_sharpe = []
            result = ana.analysis(temp_df.loc[:, self.best_param])
            self.best_sharpe = result['夏普比率'][0]

        n = np.float(self.best_param)

        if len(self.his_data['close']) < n or self.best_freq < 1.2:
            self.last_signal.append([0, self.his_data['time']])
            self.target_position(0, self.his_data['last'])
            return

        signal = np.sign(self.his_data['close'][-1] - np.mean(self.his_data['close'][-int(n):]))
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])





pre_date = 180  # 参数寻优长度
trade_date = 90  # 回测长度
num_all = 2  # 最优组合数
Base_name = 'MA'
symbol_name = '螺纹'
symbol = 'rb'
slip_point = 0  # 滑点
comission_rate = 0.0001
min_point = 1  # 最小变动价格
multip = 10  # 交易乘数
start_date = '2013-01-01'
end_date = '2018-01-01'
# end_date='2021-09-01'
start_date = pd.to_datetime(start_date) + datetime.timedelta(pre_date+200)
# for freq in ['1min', '5min', '15min', '30min', '60min', '1d']:
for freq in ['30min', '60min', '1d']:
    stragety_name = Base_name + '_' + freq  # 策略名
    filedir = './result/' + symbol_name + '/'  # 图片保存地址
    pic_name = symbol + '_' + stragety_name + "wfo"  # 图片名称
    jz_dir = filedir + stragety_name + '/净值.xlsx'

    signal_df = pd.DataFrame()
    jz_df = pd.DataFrame()
    result_df = pd.DataFrame()
    for best_param_num in range(1, num_all+1):
        roc = MA(start_date=start_date,
                 end_date=end_date,
                 comission_rate=comission_rate,
                 slip_point=slip_point,
                 min_point=min_point,
                 init_cash=10000000,
                 main_file='./行情数据库/' + symbol_name + '/',
                 symbol=symbol.upper(),
                 multip=multip,  # 交易乘数
                 freq=freq,
                 cal_way='open')
        roc.param_add(pre_date, trade_date, jz_dir, best_param_num=best_param_num)
        roc.run()
        roc.jz_plot(pic_name+'_'+str(best_param_num), filedir+stragety_name+'/')
        result = roc.analysis()
        result['参数'] = best_param_num
        signal = roc.signal
        result = result.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])

        if len(signal_df) == 0:
            signal_df = signal
            result_df = result
            jz_df = roc.jz
        else:
            signal_df = pd.concat([signal_df, signal], axis=1)
            jz_df = pd.concat([jz_df, roc.jz], axis=1)
            result_df = pd.concat([result_df, result], axis=0)
    signal_df.columns = range(1, num_all+1)
    jz_df.columns = range(1, num_all+1)
    signal_df.to_excel(filedir+stragety_name+'/wfo_最优两组参数signal.xlsx')
    result_df.to_excel(filedir+stragety_name+'/wfo_最优两组参数的绩效.xlsx')
    jz_df.to_excel(filedir+stragety_name+'/wfo_最优两组参数的净值.xlsx')

    cmb = Cmb(start_date=start_date,
              end_date=end_date,
              comission_rate=comission_rate,
              slip_point=slip_point,
              min_point=min_point,
              init_cash=10000000,
              main_file='./行情数据库/' + symbol_name + '/',
              symbol=symbol.upper(),
              multip=multip,  # 交易乘数
              freq=freq,
              cal_way='open')
    cmb.param_add(jz_dir=filedir+stragety_name+'/wfo_最优两组参数signal.xlsx')
    cmb.run()
    cmb.jz_plot(pic_name+'_final', filedir+stragety_name+'/')
    result = cmb.analysis()
    result['参数'] = 'final'
    signal = cmb.signal
    result = result.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
    final_jz = cmb.jz
    signal.to_excel(filedir+stragety_name+'/wfo_组合signal.xlsx')
    result.to_excel(filedir+stragety_name+'/wfo_组合绩效.xlsx')
    final_jz.to_excel(filedir+stragety_name+'/wfo_组合净值.xlsx')