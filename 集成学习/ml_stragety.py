import pandas as pd
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import math


def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance


spy = pd.read_csv(r"./data/RB_1d.csv")
spy = spy.set_index(spy.columns[0])
spy.index = pd.to_datetime(spy.index, format="%Y-%m-%d")
spy = spy.loc['2016':]
# spy = spy.sort_values(['Date'])


spy_c = spy['收盘价']

tlen = 30
hold = 10
thred = 2

def series_cut(series, tlen, hold):
    '''根据回溯期和持有期将序列拆分'''
    tseries = []
    for i in range(tlen, len(series) - hold):
        sig = (series.iloc[i - tlen: i] / series.iloc[i]).values
        ret = series.iloc[i+hold] / series.iloc[i] - 1
        tseries.append([sig, ret])
    return tseries

def dist_cal(test_series, target_series):
    '''将目标序列和测试序列进行比较，计算相似度'''
    dist_list = []
    for num, i in enumerate(test_series):
        try:
            distance, path = fastdtw(i[0], target_series, dist=euclidean)
        except:
            print('errer: '+str(num))
        dist_list.append([distance, i[1]])
    return dist_list
    pass

def signal_cal(dist_list, thred, weight='weighted'):
    '''计算交易信号'''
    dist_frame = pd.DataFrame(dist_list, columns=['Dist', 'Ret']).sort_values(by='Dist')
    pick_dist = dist_frame[dist_frame.Dist < thred]
    ratio = len(pick_dist[pick_dist['Ret'] > 0]) / len(pick_dist[pick_dist['Ret'] < 0])
    if ratio > 0.55:
        return 1
    elif 1 - ratio <0.55:
        return -1
    else:
        return 0
    # if weight == 'equal':
    #     return pick_dist['Ret'].mean()
    # elif weight == 'weighted':
    #     pick_dist['Dist'] = 1 / pick_dist['Dist']
    #     w = pick_dist['Dist'] / pick_dist['Dist'].sum()
    #     return (w * pick_dist['Ret']).sum()


df = []
for i in range(tlen*10, len(spy_c) - hold, hold):
    target_series = spy_c.iloc[i - tlen: i]
    target_series = (target_series / target_series.iloc[0]).values
    test_series = spy_c.iloc[: i - tlen]
    test_series = series_cut(test_series, tlen, hold)
    dist_list = dist_cal(test_series, target_series)
    estimate_ret = signal_cal(dist_list, thred, 'equal')
    real_ret = spy_c.iloc[i + hold] / spy_c.iloc[i] - 1
    df.append([estimate_ret, real_ret])
    print('sum: '+str(len(spy_c) - hold), 'now: '+str(i))
ddf = pd.DataFrame(df, columns=['estimate_ret', 'real_ret'])
ddf.to_csv('df.csv')


ddf['signal'] = ddf['estimate_ret'].apply(lambda x: 1 if x > 0 else -1)
ddf['stragety_ret'] = ddf['real_ret'] * ddf['signal']

ddf['stragety_jz'] = (1 + ddf['stragety_ret']).cumprod()
ddf['real_jz'] = (1 + ddf['real_ret']).cumprod()

ddf[['stragety_jz', 'real_jz']].plot()
plt.show()