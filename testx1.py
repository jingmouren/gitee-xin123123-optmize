import pandas as pd
import numpy as np



data = pd.read_excel(r"C:\Users\Administrator\Desktop\3.2策略动态权益.xlsx")
data = data.set_index(data.columns[0])
ret = (data.diff()/data.shift()).fillna(0)
std_20 = ret.rolling(20).std().dropna()
def f(x, n):
    x = x.sort_values()
    return x[int(len(x) * n)]

# 日波动率
a_90 = std_20.apply(f, n=0.9)
a_95 = std_20.apply(f, n=0.95)
a_99 = std_20.apply(f, n=0.99)
aa = pd.concat([a_90, a_95, a_99], axis=1)
aa.columns = ['90分位', '95分位', '99分位']
aa['90分位'] = aa['90分位'].map(lambda x: format(x, '.2%'))
aa['95分位'] = aa['95分位'].map(lambda x: format(x, '.2%'))
aa['99分位'] = aa['99分位'].map(lambda x: format(x, '.2%'))
aa = aa.sort_index(ascending=False)
aa.to_csv('aa.csv')


# 预期年化收益范围 预期年化收益
rolling_ret = ret.rolling(250).mean().dropna()*250
b_05 = rolling_ret.apply(f, n=0.05) * 0.6
b_50 = rolling_ret.apply(f, n=0.50) * 0.6
b_90 = rolling_ret.apply(f, n=0.90) * 0.6
bb = pd.concat([b_05, b_50, b_90], axis=1)
bb.columns = ['05分位', '50分位', '90分位']
bb['05分位'] = bb['05分位'].map(lambda x: format(x, '.2%'))
bb['90分位'] = bb['90分位'].map(lambda x: format(x, '.2%'))
bb['50分位'] = bb['50分位'].map(lambda x: format(x, '.2%'))
bb['范围'] = bb.apply(lambda x: str(x['05分位']) + '-' + str(x['90分位']), axis=1)
bb = bb.sort_index(ascending=False)
bb.to_csv('bb.csv')

# 年化波动率范围
std_60 = ret.rolling(60).std().dropna()

c_05 = std_20.apply(f, n=0.05)*np.sqrt(250)
c_95 = std_20.apply(f, n=0.95)*np.sqrt(250)

cc = pd.concat([c_05, c_95], axis=1)
cc.columns = ['05分位', '95分位']
cc['05分位'] = cc['05分位'].map(lambda x: format(x, '.2%'))
cc['95分位'] = cc['95分位'].map(lambda x: format(x, '.2%'))
cc['范围'] = cc.apply(lambda x: str(x['05分位']) + '-' + str(x['95分位']), axis=1)
cc = cc.sort_index(ascending=False)
cc.to_csv('cc.csv')


# 净持仓风险率
net_m = pd.read_excel(r"C:\Users\Administrator\Desktop\3.2净持仓风险率.xlsx")
net_m = net_m.set_index(net_m.columns[0]).abs()
d_50 = net_m.apply(f, n=0.5)
d_95 = net_m.apply(f, n=0.95)

dd = pd.concat([d_50, d_95], axis=1)
dd.columns = ['50分位', '95分位']
dd['50分位'] = dd['50分位'].map(lambda x: format(x, '.2%'))
dd['95分位'] = dd['95分位'].map(lambda x: format(x, '.2%'))
dd['范围'] = dd.apply(lambda x: str(x['50分位']) + '-' + str(x['95分位']), axis=1)
dd = dd.sort_index(ascending=False)
dd.to_csv('dd.csv')


# 高点回撤
trace = pd.read_excel(r"C:\Users\Administrator\Desktop\3.2高点回撤.xlsx")
trace = trace.set_index(trace.columns[0]).abs()
e_70 = trace.apply(f, n=0.7)
e_95 = trace.apply(f, n=0.95)
e_99 = trace.apply(f, n=0.99)

ee = pd.concat([e_70, e_95, e_99], axis=1)
ee.columns = ['70分位', '95分位', '99分位']
ee['70分位'] = ee['70分位'].map(lambda x: format(x, '.2%'))
ee['95分位'] = ee['95分位'].map(lambda x: format(x, '.2%'))
ee['99分位'] = ee['99分位'].map(lambda x: format(x, '.2%'))
ee['范围70~95'] = ee.apply(lambda x: str(x['70分位']) + '-' + str(x['95分位']), axis=1)
ee['范围95~99'] = ee.apply(lambda x: str(x['95分位']) + '-' + str(x['99分位']), axis=1)
ee = ee.sort_index(ascending=False)
ee.to_csv('ee.csv')

