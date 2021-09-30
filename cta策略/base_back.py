import pandas as pd
from cta策略.Base.backtest import SimpleBacktest
import talib as ta
import itertools
from cta策略.Stragety_factory import *




n_list1 = [[5,10,20,40,60,80,100,120,140,160,180],[0.5,1,1.5,2,3]]  # 参数
Base_name1 = 'Aberration'

n_list2 = [[2,4,6,16],[3,6,12,24],[4,8,16,32],[5,10,20,40],[6,12,24,48]]  # 参数
Base_name2 = 'BBI'

n_list3 = [[5,10,20],[1,2,3,4,5,6,7,8,9,10]]  # 参数
Base_name3 = 'BIAS'
# #
n_list4 = [[10,20,30, 40,50,60,80,100,120],[0.5,1,1.5,2,2.5,3]]  # 参数
Base_name4 = 'BOLL'
# #
n_list5 = [3,5,9,10,12,14,16,20,30,40,50,60,80,100,120]  # 参数
Base_name5 = 'CCI'
# #
n_list6 = [3,5,9,10,12,14,16,20,30,40,50,60,80,100,120]  # 参数
Base_name6 = 'CMO'
# #
n_list7 = [[5,25],[10,50],[12,60],[14,70],[20,100]]  # 参数
Base_name7 = 'DMA'
# #
n_list8 = [[4,2,2],[9,3,3],[16,4,4],[25,5,5],[36,6,6], [49,7,7],[64,8,8],[81,9,9],[100,10,10]]  # 参数
Base_name8 = 'KDJ'
# # #
n_list9 = [5, 10, 20, 30, 40,50, 60,70, 80,90, 100,110, 120]  # 参数
Base_name9 = 'MA'
#
n_list10 = [[7,14,5],[8,16,6],[9,18,7],[10,20,8],[12,26,9],[13,26,10]]  # 参数
Base_name10 = 'MACD'
# #
n_list11 = [3,5,9,10,12,14,20,60]  # 参数
Base_name11 = 'ROC'
# #
n_list12 = [2,3,5,6,7,8,9,10,12,14,15,16,17,18,19,20]  # 参数
Base_name12 = 'RSI'
# #
n_list13 = [[5,10,15,20,30,40,60,80], [10,15,20,30,40,60,80,120,140,160]]  # 参数
Base_name13 = 'SMA'
# #
n_list14 = [[5,6,7,8,9,10,11,12,14,16,18,20], [4,5,6,7,8,9,10]]  # 参数
Base_name14 = 'TRIX'

n_list_all = [n_list1, n_list2, n_list3, n_list4, n_list5, n_list6, n_list7, n_list8, n_list9, n_list10, n_list11
            , n_list12, n_list13, n_list14]
Base_name_all = [Base_name1, Base_name2, Base_name3, Base_name4, Base_name5, Base_name6, Base_name7, Base_name8
             , Base_name9, Base_name10, Base_name11, Base_name12, Base_name13, Base_name14]


symbol_name = '白银'
symbol = 'ag'

symbol_name = '白银'
symbol = 'ag'

symbol_name = '甲醇'
symbol = 'ma'

symbol_name = '豆一'
symbol = 'a'

symbol_name = 'PP'
symbol = 'pp'

symbol_name = 'PTA'
symbol = 'ta'

symbol_name = 'PVC'
symbol = 'v'

symbol_name = '螺纹'
symbol = 'rb'

symbol_name = '塑料'
symbol = 'l'

symbol_name = '棕榈'
symbol = 'p'

symbol_name = '橡胶'
symbol = 'ru'


slip_point = 1  # 滑点
comission_rate = 0.0005
min_point = 1  # 最小变动价格
multip = 10  # 交易乘数
start_date = '2013-01-01'
end_date = '2018-01-01'
# end_date='2021-09-01'
for num in range(len(n_list_all)):
    n_list = n_list_all[num]
    Base_name = Base_name_all[num]
    # for freq in ['1min', '5min', '15min', '30min', '60min', '1d']:
    for freq in ['1d']:
        stragety_name = Base_name + '_' + freq  # 策略名
        filedir = './result/' + symbol_name + '/'  # 图片保存地址
        pic_name = symbol + '_' + stragety_name + "_参数："  # 图片名称
        result_df = pd.DataFrame()
        jz_df = pd.DataFrame()
        name_list = []
        if len(n_list) <=2:
            nn_list = list(itertools.product(*n_list))
        else:
            nn_list = n_list
        for n in nn_list:
            if type(n) != int:
                n = list(n)
            roc = eval(Base_name)(start_date=start_date,
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
            roc.jz_plot(fig_name, filedir+stragety_name+'/', is_show=False)
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
        signal.columns = name_list
        signal.to_excel(filedir + stragety_name + '/各参数signal.xlsx')
        print(result_df)