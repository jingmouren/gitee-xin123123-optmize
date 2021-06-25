import pandas as pd
from Base import vector_backtest
from formula_package import boll_signal, ta
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
from scipy.special import comb, perm
import itertools
from tqdm import tqdm




def pbo_cal(file_name):
    jz = pd.read_csv(file_name)
    jz = jz.set_index(jz.columns[0])
    jz.index = pd.to_datetime(jz.index, format="%Y-%m-%d")
    num = 96  # 数据分段数量
    s = 16  # 子集的数量
    half_s = int(s/2)  # 样本内外子集数量（子集的一半）
    s_list = list(range(1, s + 1))
    c = int(comb(s, half_s))
    jz_96 = jz.iloc[::int(len(jz.index) / num), :]
    ret = (jz_96.diff() / jz_96.shift()).dropna()

    w_list = np.zeros(c)
    for num, i in enumerate(itertools.combinations(s_list, half_s)):
        inner_m = ret.iloc[list(i), :]
        outer_m = ret.iloc[sorted(set(s_list) - set(i)), :]
        inner_sharpe = (inner_m.mean() / inner_m.std()).dropna()
        inner_name = inner_sharpe.sort_values().index[-1]
        outer_sharpe = (outer_m.mean() / outer_m.std()).dropna()
        outer_rank = (len(outer_sharpe) - outer_sharpe.rank() + 1)/(len(outer_sharpe) + 1)
        w = outer_rank[inner_name]
        w_list[num] = w
        print('\r process:{}%'.format(num/c*100), end='')
    pbo = len(w_list[w_list>0.5])/len(w_list)
    print(len(ret))
    print(pbo)
#
#



def para_pick(file_name):
    jz = pd.read_csv(file_name)
    jz = jz.set_index(jz.columns[0])
    jz.index = pd.to_datetime(jz.index, format="%Y-%m-%d")
    num = 96  # 数据分段数量
    s = 12  # 子集的数量
    half_s = int(s/2)  # 样本内外子集数量（子集的一半）
    s_list = list(range(1, s + 1))
    c = int(comb(s, half_s))
    jz_96 = jz.iloc[::int(len(jz.index) / num), :]
    ret = (jz_96.diff() / jz_96.shift()).dropna()

    w_list = pd.DataFrame()
    for num, i in enumerate(itertools.combinations(s_list, half_s)):
        inner_m = ret.iloc[list(i), :]
        outer_m = ret.iloc[sorted(set(s_list) - set(i)), :]
        inner_sharpe = (inner_m.mean() / inner_m.std()).dropna()
        inner_rank = (len(inner_sharpe) - inner_sharpe.rank() + 1)/(len(inner_sharpe) + 1)
        outer_sharpe = (outer_m.mean() / outer_m.std()).dropna()
        outer_rank = (len(outer_sharpe) - outer_sharpe.rank() + 1) / (len(outer_sharpe) + 1)
        ratio = 2 * abs(inner_rank - outer_rank) / (len(inner_rank) + len(outer_rank))
        if len(w_list) == 0:
            w_list = ratio
        else:
            w_list = pd.concat([w_list, ratio], axis=1, sort=True)
        # outer_rank = (len(outer_sharpe) - outer_sharpe.rank() + 1)/(len(outer_sharpe) + 1)
        # w = outer_rank[inner_name]
        print('\r process:{}%'.format(num/c*100), end='')
    ratio_mean = w_list.mean(axis=1)
    sharpe_all = ret.mean() / ret.std()
    ratio_mean = pd.concat([ratio_mean, sharpe_all], axis=1)
    ratio_mean.columns = ['ratio_mean', 'sharpe']
    ratio_mean.to_csv('ratio_mean.csv')
    print(ratio_mean)


# file_name = r'bolling_jz.csv'
# pbo_cal(file_name)

file_name = r'bolling_jz.csv'
para_pick(file_name)
