import pandas as pd
from cta策略.Base.backtest import SimpleBacktest
import talib as ta

class KDJ(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.short, self.long, self.mmid = n
    def signal_cal(self):
        n1 = self.short
        n2 = self.long
        n3 = self.mmid

        if len(self.his_data['close']) < n1 + n2+n3:
            self.target_position(0, self.his_data['last'])
            return
        slowk, slowd = ta.STOCH(self.his_data['high'],
                                self.his_data['low'],
                                self.his_data['close'],
                                fastk_period=n1,
                                slowk_period=n2,
                                slowk_matype=0,
                                slowd_period=n3,
                                slowd_matype=0)
        K = slowk[-1]
        D = slowd[-1]
        J = (3 * K) - (2 * D)
        signal = 0
        if K > D:
            signal = 1
        if K < D:
            signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass



n_list = [[4,2,2],[9,3,3],[16,4,4],[25,5,5],[36,6,6]]  # 参数
n_list = [[4,2,2],[9,3,3],[16,4,4],[25,5,5],[36,6,6], [49,7,7],[2,2,2],[64,8,8],[81,9,9],[3,3,3],[4,4,4]
          ,[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9]]  # 参数
Base_name = 'KDJ'
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
# for freq in ['30min', '60min', '1d']:
for freq in ['1d']:
    stragety_name = Base_name + '_' + freq  # 策略名
    filedir = './result/' + symbol_name + '/'  # 图片保存地址
    pic_name = symbol + '_' + stragety_name + "_参数："  # 图片名称
    result_df = pd.DataFrame()
    jz_df = pd.DataFrame()
    name_list = []
    signal = pd.DataFrame()
    for n in n_list:
        roc = KDJ(start_date=start_date,
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
