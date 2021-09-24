import pandas as pd
import numpy as np
from cta策略.Base.backtest import SimpleBacktest
import talib as ta

class DMA(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
        self.dma = []
        self.ama = []
    def param_add(self, n):
        # 参数设置
        self.n1, self.n2 = n

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def signal_cal(self):
        n1 = self.n1
        n2 = self.n2

        if len(self.his_data['close']) < n2+n1:
            self.target_position(0, self.his_data['last'])
            return

        sma = np.mean(self.rolling_window(self.his_data['close'], n1), -1)
        lma = np.mean(self.rolling_window(self.his_data['close'], n2), -1)
        dma1 = sma[-1] - lma[-1]
        dma2 = sma[-2] - lma[-2]

        ama1 = np.mean(sma[-n1:] - lma[-n1:])
        ama2 = np.mean(sma[-n1 - 1:-1] - lma[-n1 - 1:-1])
        signal = 0
        if dma1 > 0 and dma1 - dma2 > 0 and ama1 > 0 and ama1 - ama2 > 0:
            signal = 1
        elif dma1 < 0 and dma1 - dma2 < 0 and ama1 < 0 and ama1 - ama2 < 0:
            signal = -1
        elif self.last_hands > 0:
            signal = 1
        elif self.last_hands < 0:
            signal = -1

        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass



n_list = [[5,25],[10,50],[12,60],[14,70],[20,100]]  # 参数
Base_name = 'DMA'
symbol_name = '螺纹'
symbol = 'rb'
slip_point = 0  # 滑点
comission_rate = 0.0001
min_point = 1  # 最小变动价格
multip = 10  # 交易乘数
start_date = '2013-01-01'
end_date = '2018-01-01'
# end_date='2021-09-01'
# for freq in ['1min', '5min', '15min', '30min', '60min', '1d']:
for freq in ['30min', '60min', '1d']:
    stragety_name = Base_name + '_' + freq  # 策略名
    filedir = './result/' + symbol_name + '/'  # 图片保存地址
    pic_name = symbol + '_' + stragety_name + "_参数："  # 图片名称
    result_df = pd.DataFrame()
    jz_df = pd.DataFrame()
    name_list = []
    for n in n_list:
        roc = DMA(start_date=start_date,
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

        fig_name = pic_name + str(n)
        name_list.append(str(n))
        roc.param_add(n)
        roc.run()
        roc.jz_plot(fig_name, filedir+stragety_name+'/')
        result = roc.analysis()
        result['参数'] = str(n)
        print(result)
        if len(result_df) == 0:
            result_df = result
            jz_df = roc.jz
        else:
            result_df = pd.concat([result_df, result], axis=0, sort=True)
            jz_df = pd.concat([jz_df, roc.jz], axis=1, sort=True)
    result_df = result_df.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
    result_df.to_excel(filedir+stragety_name+'/绩效.xlsx')
    jz_df.columns = name_list
    jz_df.to_excel(filedir+stragety_name+'/净值.xlsx')
    print(result_df)




