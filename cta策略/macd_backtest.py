import pandas as pd
from cta策略.Base.backtest import SimpleBacktest
import talib as ta

class MACD(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.short, self.long, self.mmid = n
    def signal_cal(self):
        short = self.short
        long = self.long
        mmid = self.mmid

        if len(self.his_data['close']) < long + mmid:
            self.target_position(0, self.his_data['last'])
            return
        macd, signal, hist = ta.MACD(self.his_data['close'], fastperiod=short, slowperiod=long, signalperiod=mmid)
        signal = 0
        if macd[-1] > 0:
            signal = 1
        if macd[-1] < 0:
            signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass



n_list = [[7,14,5],[8,16,6],[9,18,7],[10,20,8],[12,26,9],[13,26,10]]  # 参数
Base_name = 'MACD'
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
    signal = pd.DataFrame()
    for n in n_list:
        roc = MACD(start_date=start_date,
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
            signal = roc.signal
        else:
            result_df = pd.concat([result_df, result], axis=0, sort=True)
            jz_df = pd.concat([jz_df, roc.jz], axis=1, sort=True)
            signal = pd.concat([signal, roc.signal], axis=1, sort=True)
    result_df = result_df.reindex(columns=['参数', '年化收益率', '夏普比率', '卡玛比率', '季度胜率'])
    result_df.to_excel(filedir+stragety_name+'/绩效.xlsx')
    jz_df.columns = name_list
    jz_df.to_excel(filedir+stragety_name+'/净值.xlsx')
    signal.columns = [str(x) for x in n_list]
    signal.to_excel(filedir + stragety_name + '/各参数signal.xlsx')
    print(result_df)

