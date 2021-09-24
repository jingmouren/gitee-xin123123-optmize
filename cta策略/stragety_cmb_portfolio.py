import pandas as pd
import numpy as np
from analysis_model import Analysis
import datetime
from cta策略.Base.backtest import SimpleBacktest, Cmb
import talib as ta




class Portfolio(Cmb):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, jz_dir_list):
        # 参数设置
        signal_df = pd.DataFrame()
        for jz_dir in jz_dir_list:
            signal = pd.read_excel(jz_dir)
            signal = signal.set_index(signal.columns[0]).fillna(0).mean(axis=1)
            if len(signal_df) == 0:
                signal_df = signal
            else:
                signal_df = pd.concat([signal_df, signal], axis=1)
        signal.index = pd.to_datetime(signal.index)
        signal_df = signal_df.fillna(0).mean(axis=1)
        self.signal = signal_df

    def signal_cal(self):
        time = self.his_data['time']
        if time not in self.signal.index:
            self.target_position(0, self.his_data['last'])
            return
        signal = self.signal.loc[time]
        if signal > 0:
            signal = 1
        if signal < 0:
            signal = -1
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        self.last_signal.append([signal, time])



Base_name = 'Portfolio'
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
    cmb.param_add(jz_dir_list=jz_dir_list)
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