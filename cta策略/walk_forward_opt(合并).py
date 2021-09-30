import pandas as pd
import numpy as np
from analysis_model import Analysis
import datetime
from cta策略.Base.backtest import SimpleBacktest
import talib as ta
import statsmodels.api as sm

class Cmb(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, pre_date, trade_date, jz_dir):
        # 参数设置
        self.pre_date = pre_date
        self.trade_date = trade_date
        base_jz = pd.read_excel(jz_dir)
        base_jz = base_jz.set_index(base_jz.columns[0])
        self.base_jz = base_jz
        # 信号表
        signal_dir = '/'.join(jz_dir.split('/')[:-1] + ['各参数signal.xlsx'])
        base_signal = pd.read_excel(signal_dir)
        self.base_signal = base_signal.set_index('time')

        if self.start_date > max(self.base_jz.index):
            print('单策略数据长度不够！！！')
            print('last history time: ', max(self.base_jz.index))
            print('start time: ', self.start_date)
            exit()

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def signal_cal(self):
        if len(self.base_jz.loc[:self.his_data['time'], :].iloc[:-1, :].index) < self.pre_date+1:
            self.target_position(0, self.his_data['last'])
            return
        if (self.num - 1) % self.trade_date == 0:  # 每隔n个周期计算最优参数
            pass
            time = self.his_data['time']
            temp_df_all = self.base_jz.loc[:time, :].iloc[-self.pre_date:-1, :]
            temp_df_all = self.base_jz.loc[:time, :].iloc[:-1, :]

            test_n = int(len(temp_df_all.index) * 0.7)
            yz_n = len(temp_df_all.index) - test_n
            temp_df = temp_df_all

            ret = (temp_df.diff() / temp_df.shift()).dropna()
            def f(x, pre_date):
                if len(x) == 0:
                    return 0

                # long_x = x
                # short_x = x.iloc[-pre_date:]
                # long_y = np.array((long_x+1).cumprod())
                # long_x = np.array(range(1, 1 + len(long_y)))
                # X = sm.add_constant(long_x)
                # model = sm.OLS(long_y, X)
                # results = model.fit()
                # long_t = results.tvalues[1]
                #
                # short_y = np.array((short_x + 1).cumprod())
                # short_x = np.array(range(1, 1 + len(short_y)))
                # X = sm.add_constant(short_x)
                # model = sm.OLS(short_y, X)
                # results = model.fit()
                # short_t = results.tvalues[1]
                #
                # return np.sign(short_t - long_t)
                # if len(x[x != 0]) == 0:
                #     return 0
                # return -x.std()
                # return x.mean()/x.std()
                # x = (x + 1).cumprod()
                # return -max(1 - x/x.cummax())
                long_x = x
                short_x = x.iloc[-pre_date:]
                short_sharpe = short_x.mean() / (short_x.std()+0.0000000001)
                long_sharpe = long_x.mean() / (long_x.std()+0.0000000001)
                long_win = long_x[long_x > 0].sum()
                long_loss = abs(long_x[long_x < 0].sum())+0.000001
                long_ykb = long_win / long_loss
                short_win = short_x[short_x > 0].sum()
                short_loss = abs(short_x[short_x < 0].sum())+0.000001
                short_ykb = short_win / short_loss
                return long_ykb

            factor = pd.Series(index=ret.columns).fillna(0)

            freq_ret = ret.apply(f, args=[self.pre_date])
            freq_ret = freq_ret[freq_ret != 0]
            # freq_ret = freq_ret[freq_ret > 1]

            freq_ret = freq_ret.sort_values()
            # freq_ret = pd.Series(range(len(freq_ret)), index=freq_ret.index)
            # freq_ret = freq_ret.loc[cor_g.index[0:int(len(cor_g)/2)]]
            # freq_ret = freq_ret.loc[cor_g.index[-int(len(cor_g)/2):]]
            self.buy_name = freq_ret[freq_ret>5].index
            self.sell_name = freq_ret[freq_ret<-5].index
            # factor = freq_ret.iloc[-3:]
            # self.factor = factor[factor>1]
            # freq_ret = factor.sort_values()
            # print(freq_ret)

            # ratio = (freq_ret - freq_ret.min()) / (freq_ret.max() - freq_ret.min())
            # ratio = (freq_ret - freq_ret.mean()) / abs(freq_ret.std())
            # sign_ratio = np.sign(ratio)
            # ratio = ratio / abs(ratio).sum()
            # ratio = ratio * sign_ratio

            self.ratio = freq_ret
            # self.freq_ret = freq_ret
            # self.chg = pd.Series(freq_ret, index=freq_ret.index).fillna(1)
            # self.chg[freq_ret[freq_ret < 1].index] = -1





        time = self.his_data['time']
        today_signal = self.base_signal.loc[time, :]
        buy_signal = (today_signal.loc[self.buy_name].sum())
        sell_signal = (today_signal.loc[self.sell_name].sum())
        signal = (self.ratio * today_signal).sum()
        # signal = (buy_signal + sell_signal).sum()
        if abs(signal) > 0:
            signal = np.sign(signal)
        else:
            signal = 0

        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])


name_list = ['Aberration',
            'BBI',
            'BIAS',
            'BOLL',
            'CCI',
            'CMO',
            'DMA',
            'KDJ',
            'MA',
            'MACD',
            'ROC',
            'RSI',
            'SMA',
            'TRIX']

pre_date = 90  # 参数寻优长度
trade_date = 90  # 回测长度


# symbol_name = '白银'
# symbol = 'ag'
#
symbol_name = '甲醇'
symbol = 'ma'

symbol_name = '豆一'
symbol = 'a'

# symbol_name = 'PP'
# symbol = 'pp'

# symbol_name = 'PTA'
# symbol = 'ta'

# symbol_name = 'PVC'
# symbol = 'v'

# symbol_name = '螺纹'
# symbol = 'rb'

symbol_name = '塑料'
symbol = 'l'

symbol_name = '橡胶'
symbol = 'ru'

slip_point = 1  # 滑点
comission_rate = 0.0005
min_point = 1  # 最小变动价格
multip = 10  # 交易乘数
start_date = '2013-01-01'
end_date = '2018-01-01'
# end_date='2021-09-01'
start_date = pd.to_datetime(start_date) + datetime.timedelta(pre_date+200)

for Base_name in name_list:
    print('策略：', Base_name)
    # for freq in ['1min', '5min', '15min', '30min', '60min', '1d']:
    for freq in ['1d']:
        stragety_name = Base_name + '_' + freq  # 策略名
        filedir = './result/' + symbol_name + '/'  # 图片保存地址
        pic_name = symbol + '_' + stragety_name + "wfo"  # 图片名称
        jz_dir = filedir + stragety_name + '/净值.xlsx'

        signal_df = pd.DataFrame()
        jz_df = pd.DataFrame()
        result_df = pd.DataFrame()

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
        cmb.param_add(pre_date, trade_date, jz_dir)
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

