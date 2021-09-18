import pandas as pd
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import math


def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance


def get_stats(s, n=252):
    cnt = len(s)
    wins = len(s[s > 0])
    losses = len(s[s < 0])
    mean_trd = round(s.mean(), 3)
    sd = round(np.std(s), 3)
    sharpe_r = round((s.mean() / np.std(s)) * np.sqrt(n), 4)
    print('Trades:', cnt, \
    '\nWins:', wins, \
    '\nLosses:', losses, \
    '\nMean:', mean_trd, \
    '\nStd Dev:', sd, \
    '\nSharpe Ratio:', sharpe_r)



spy = pd.read_csv(r"./data/RB_1d.csv")
spy = spy.set_index(spy.columns[0])
spy.index = pd.to_datetime(spy.index, format="%Y-%m-%d")
# spy = spy.sort_values(['Date'])


spy_c = spy['收盘价']

tlen = 5
hold = 7
thre = 5

tseries = []
for i in range(tlen, len(spy_c) - hold - 1, tlen):
    pctc = spy_c.iloc[i - tlen:i].pct_change()[1:].values * 100
    res = (spy_c[i + hold + 1] - spy_c[i]) / spy_c[i] * 100
    tseries.append((pctc, res))


dist_pairs = []
for i in range(len(tseries)):
    for j in range(len(tseries)):
        print(i, j)
        dist = dtw_dist(tseries[i][0], tseries[j][0])
        dist_pairs.append((i, j, dist, tseries[i][1], tseries[j][1]))


dist_frame = pd.DataFrame(dist_pairs, columns=['A', 'B', 'Dist', 'A Ret', 'B Ret'])
sf = dist_frame[dist_frame['Dist'] > 0].sort_values(['A', 'B']).reset_index(drop=1)
sfe = sf[sf['A'] + math.ceil(float(tlen + hold) / tlen) <= sf['B']]
winf = sfe[(sfe['Dist'] <= thre) & (sfe['A Ret'] > 0)]

excluded = {}
return_list = []


def get_returns(r):
    if excluded.get(r['A']) is None:
        return_list.append(r['B Ret'])
        if r['B Ret'] < 0:
            excluded.update({r['A']: 1})


winf.apply(get_returns, axis=1)
get_stats(pd.Series(return_list))
aa = pd.concat([pd.Series(return_list), spy_c], axis=1)
bb = (pd.Series(return_list)/100+1).cumprod()
bb.plot()
plt.show()
aa.columns= ['stragety', 'init']
aa.plot()