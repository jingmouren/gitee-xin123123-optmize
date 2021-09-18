import pandas as pd
from Base import vector_backtest
from formula_package import *
import matplotlib.pyplot as plt
from data_simulate import DataSim
import numpy as np
import math
from scipy.signal import savgol_filter

if __name__ == "__main__":
    start_date = '20100101'
    end_date = '20200601'

    file_dir = r"./data/RB_1d.csv"
    sum1 = DataSim(start_date, end_date, file_dir)
    sum1.relative_cal()
    sim_data = sum1.random_cal()
    sim_data.to_csv('./data/sim_RB.csv')
    a = pd.DataFrame(sim_data['Close'])

    #### 样本外模拟测试 #########

    file_dir = r"./data/sim_RB.csv"
    test1 = vector_backtest(start_date, end_date, file_dir, freq='1d', cal_way='open')

    timeperiod = 160
    std = 0.5

    print('timeperiod: ' + str(timeperiod) + '; std: ' + str(std))
    signal = boll_signal(test1.data['Close'], timeperiod, std)
    test1.add_stragety(signal=signal)
    test1.run()
    data = pd.concat([test1.basejz, test1.jz, test1.singal], axis=1)
    # data.columns = []
    test1.jz_plot()
    aa = test1.jz
    ma_60_ = test1.jz.rolling(60).mean()#.dropna()


    ma_60 = pd.Series(savgol_filter(test1.jz, 51, 3), index=test1.jz.index)  # window size 51, polynomial order 3

    diff_ma = ma_60.diff().fillna('drop')
    diff_ma = np.array(diff_ma)
    max_point = [0]
    min_point = [0]
    for d in range(1, len(diff_ma) - 1):
        if diff_ma[d] == 'drop':
            continue
        if diff_ma[d-5] == 'drop':
            continue
        if diff_ma[d] > 0 and diff_ma[d+1] < 0:
            if d - min_point[-1] >= 20:
                if min_point[-1] >= max_point[-1]:
                    max_point.append(d)
                else:
                    max_point[-1] = d
        elif diff_ma[d] < 0 and diff_ma[d+1] > 0:
            if d - max_point[-1] >= 20:
                if max_point[-1] >= min_point[-1]:
                    min_point.append(d)
                else:
                    min_point[-1] = d

    max_point = test1.jz.iloc[max_point[1:]]
    min_point = test1.jz.iloc[min_point[1:]]

    # max_point = pd.Series([1]*len(max_point), index=max_point.index)
    # min_point = pd.Series([-1]*len(min_point), index=min_point.index)
    # buy_sell_point = pd.concat([max_point, min_point], axis=0)
    # data = pd.concat([test1.basejz, test1.jz, ma_60, buy_sell_point], axis=1)
    # data.columns = ['行情', '策略', '平滑策略', '高低点']
    # data['行情ret'] = data['行情'].diff() / data['行情'].shift()
    # data = data.bfill()
    # win_ret = data[data['高低点'] == 1]['行情ret']
    # loss_ret = data[data['高低点'] == -1]['行情ret']
    # win_jz = (win_ret+1).cumprod()
    # loss_jz = (loss_ret+1).cumprod()
    # win_jz.index = range(len(win_jz.index))
    # loss_jz.index = range(len(loss_jz.index))
    # w_l = pd.concat([win_jz, loss_jz], axis=1)
    # w_l.columns = ['win', 'loss']
    # w_l.plot()

    max_point = pd.Series(range(1, len(max_point)+1), index=max_point.index)
    min_point = -1 * pd.Series(range(1, len(min_point)+1), index=min_point.index)
    buy_sell_point = pd.concat([max_point, min_point], axis=0)
    data = pd.concat([test1.basejz, test1.jz, ma_60, buy_sell_point], axis=1)
    data.columns = ['行情', '策略', '平滑策略', '高低点']
    data['行情ret'] = data['行情'].diff() / data['行情'].shift()
    data = data.bfill()
    temp = data[data['高低点'] > 0]
    bb = pd.DataFrame()
    for hq in sorted(set(temp['高低点'])):
        aa = temp[temp['高低点'] == hq]['行情ret']
        if len(bb) == 0:
            bb = aa
        else:
            bb = pd.concat([bb, aa],axis=1)
    bb.columns = sorted(set(temp['高低点']))
    bb = (1+bb).cumprod()
    bb.plot()
    plt.show()
    # data.groupby(by='高低点').std()['行情ret'].plot()
    plt.show()

    cc = bb.bfill().ffill()
    dd = cc.iloc[-1, :] / cc.iloc[0, :] - 1
    print('win')
    print(dd)


    temp = data[data['高低点'] < 0]
    bb = pd.DataFrame()
    for hq in sorted(set(temp['高低点'])):
        aa = temp[temp['高低点'] == hq]['行情ret']
        if len(bb) == 0:
            bb = aa
        else:
            bb = pd.concat([bb, aa], axis=1)
    bb.columns = sorted(set(temp['高低点']))
    bb = (1 + bb).cumprod()
    bb.plot()
    plt.show()
    # data.groupby(by='高低点').std()['行情ret'].plot()
    plt.show()

    cc = bb.bfill().ffill()
    dd = cc.iloc[-1, :] / cc.iloc[0, :] - 1
    print('loss')
    print(dd)

    # print('win:', win_ret.std())
    # print('loss:', loss_ret.std())
    # data.to_csv('test.csv')
    # cummax = test1.jz.cummax()
    # cummin = test1.jz.cummin()
    # maxtrace = 1 - test1.jz / cummax
    # mintrace = 1 - cummin / test1.jz
    #
    # for d in range(len(test1.jz.index)):
    #
    #     print(d)