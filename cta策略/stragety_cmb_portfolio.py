import pandas as pd
import numpy as np
from analysis_model import Analysis
import datetime
from cta策略.Base.backtest import SimpleBacktest, Cmb
import talib as ta
import statsmodels.api as sm



class Portfolio(Cmb):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, pre_date, trade_date, jz_dir_list):
        # 参数设置
        signal_df = pd.DataFrame()
        jz_df = pd.DataFrame()
        for jz_dir in jz_dir_list:
            signal = pd.read_excel(jz_dir)


            # 净值表
            signal_dir = jz_dir.replace('signal','净值')
            base_jz = pd.read_excel(signal_dir)
            name = jz_dir.split('/')[3]
            base_jz.columns = ['time', name]
            signal.columns = ['time', name]
            base_jz = base_jz.set_index('time')
            signal = signal.set_index('time')


            if len(signal_df) == 0:
                signal_df = signal
                jz_df = base_jz
            else:
                signal_df = pd.concat([signal_df, signal], axis=1)
                jz_df = pd.concat([jz_df, base_jz], axis=1)
        signal_df.index = pd.to_datetime(signal_df.index)
        jz_df.index = pd.to_datetime(jz_df.index)
        signal_df = signal_df.fillna(0)#.mean(axis=1)
        self.base_signal = signal_df
        self.base_jz = jz_df#.dropna()
        self.pre_date = pre_date
        self.trade_date = trade_date

    def signal_cal(self):

        if (self.num - 1) % self.trade_date == 0:  # 每隔n个周期计算最优参数
            time = self.his_data['time']
            temp_df_all = self.base_jz.loc[:time, :].iloc[-self.pre_date:-1, :]
            temp_df_all = self.base_jz.loc[:time, :].iloc[:-1, :]
            if len(temp_df_all) <= 20:
                self.buy_name = []
                self.sell_name = []
                self.ratio = pd.Series(index=self.base_jz.columns).fillna(0)
                self.target_position(0, self.his_data['last'])
                return

            # temp_df_all = self.base_jz.loc[:time, :].iloc[:-1, :]
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
                short_sharpe = short_x.mean() / (short_x.std() + 0.0000000001)
                long_sharpe = long_x.mean() / (long_x.std() + 0.0000000001)
                long_win = long_x[long_x > 0].sum()
                long_loss = abs(long_x[long_x < 0].sum()) + 0.000001
                long_ykb = long_win / long_loss
                short_win = short_x[short_x > 0].sum()
                short_loss = abs(short_x[short_x < 0].sum()) + 0.000001
                short_ykb = short_win / short_loss
                return long_ykb


            factor = pd.Series(index=ret.columns).fillna(0)
            num = int(len(factor) / 5)

            freq_ret = ret.apply(f, args=[self.pre_date])
            freq_ret = freq_ret[freq_ret != 0]
            freq_ret = freq_ret.sort_values()
            # freq_ret = pd.Series(range(len(freq_ret)), index=freq_ret.index)
            # freq_ret = freq_ret.loc[cor_g.index[0:int(len(cor_g)/2)]]
            # freq_ret = freq_ret.loc[cor_g.index[-int(len(cor_g)/2):]]
            self.buy_name = freq_ret[freq_ret > 1.2].index
            self.sell_name = freq_ret[freq_ret < 0.8].index
            # freq_ret = pd.Series(range(len(freq_ret)), index=freq_ret.index)
            # freq_ret = freq_ret.loc[cor_g.index[0:int(len(cor_g)/2)]]
            # freq_ret = freq_ret.loc[cor_g.index[-int(len(cor_g)/2):]]
            # freq_ret = freq_ret[freq_ret>1]


            # num = int(num/3)
            long_ = freq_ret.iloc[-num:]
            # long_ = long_[long_>10]
            long_ratio = long_.mean()
            long_ = long_.index
            short_ = freq_ret.iloc[:num]
            # short_ = short_[short_<-10]
            short_ratio = short_.mean()
            short_ = short_.index
            ratio = (freq_ret - freq_ret.min()) / (freq_ret.max() - freq_ret.min())
            # ratio = (freq_ret - freq_ret.mean()) / abs(freq_ret.max())
            # sign_ratio = np.sign(ratio)
            # ratio = ratio / abs(ratio).sum()
            # ratio = ratio * sign_ratio

            self.ratio = ratio

            self.long_ = long_
            self.short_ = short_

        time = self.his_data['time']
        if time < min(self.base_signal.index):
            self.target_position(0, self.his_data['last'])
            return

        today_signal = self.base_signal.loc[time, :]


        # long_signal = today_signal.loc[self.long_].sum()
        # short_signal = today_signal.loc[self.short_].sum()
        #
        # signal = np.sign(long_signal - short_signal)

        # signal = (self.ratio * today_signal).sum()
        today_signal = self.base_signal.loc[time, :]
        buy_signal = (today_signal.loc[self.buy_name].sum())
        sell_signal = (today_signal.loc[self.sell_name].sum())
        signal = (self.ratio * today_signal).sum()
        # signal = (buy_signal - sell_signal).sum()
        if abs(signal) > 0:
            signal = np.sign(signal)
        else:
            signal = 0

        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])




        # time = self.his_data['time']
        # if time not in self.signal.index:
        #     self.target_position(0, self.his_data['last'])
        #     return
        # signal = self.signal.loc[time]
        # if signal > 0:
        #     signal = 1
        # if signal < 0:
        #     signal = -1
        # hands = self.capital / self.multip / self.his_data['last'] * signal
        # self.target_position(hands, self.his_data['last'])
        # self.last_signal.append([signal, time])



Base_name = 'Portfolio'


symbol_name = '白银'
symbol = 'ag'

symbol_name = '甲醇'
symbol = 'ma'

symbol_name = '豆一'
symbol = 'a'

symbol_name = 'PP'
symbol = 'pp'

# symbol_name = 'PTA'
# symbol = 'ta'

# symbol_name = 'PVC'
# symbol = 'v'

symbol_name = '螺纹'
symbol = 'rb'

symbol_name = '塑料'
symbol = 'l'


pre_date = 180
trade_date = 90

slip_point = 0  # 滑点
comission_rate = 0.001
min_point = 1  # 最小变动价格
multip = 10  # 交易乘数
start_date = '2013-01-01'
start_date = pd.to_datetime(start_date) + datetime.timedelta(pre_date+100)
end_date = '2018-01-01'
# end_date='2021-09-01'
# for freq in ['1min', '5min', '15min', '30min', '60min', '1d']:
for freq in ['1d']:
    stragety_name = Base_name + '_' + freq  # 策略名
    filedir = './result/' + symbol_name + '/'  # 图片保存地址
    pic_name = symbol + '_' + stragety_name  # 图片名称
    jz_dir = filedir + stragety_name + '/净值.xlsx'
    cmb = Portfolio(start_date=start_date,
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
    jz_dir_list = [filedir+'Aberration_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'BBI_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'BIAS_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'BOLL_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'CCI_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'CMO_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'DMA_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'KDJ_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'MA_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'MACD_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'ROC_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'RSI_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'SMA_'+freq+'/wfo_组合signal.xlsx',
                   filedir+'TRIX_'+freq+'/wfo_组合signal.xlsx',
                   ]
    cmb.param_add(pre_date, trade_date, jz_dir_list=jz_dir_list)
    cmb.run()
    cmb.jz_plot(pic_name, filedir+stragety_name+'/')
    result = cmb.analysis()
    result['参数'] = '组合'
    signal = cmb.signal
    result = result.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
    final_jz = cmb.jz
    signal.to_excel(filedir+stragety_name+'/wfo_组合signal.xlsx')
    result.to_excel(filedir+stragety_name+'/wfo_组合绩效.xlsx')
    final_jz.to_excel(filedir+stragety_name+'/wfo_组合净值.xlsx')