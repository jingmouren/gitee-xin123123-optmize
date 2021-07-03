from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_model import Analysis
from sklearn.model_selection import GridSearchCV


class PreTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        super(PreTransformer, self).__init__()

    def fit(self, X):
        self.X = X
        return self

    def data_fix(self, file_dir, cal_way):
        '''读取数据并调整格式'''
        data = pd.read_csv(file_dir)
        data = data.set_index(data.columns[0])
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
        if cal_way == 'open':
            x = data.shift(2).iloc[2:, :].dropna(how='all')  # 避免未来数据
            ret = data['开盘价'].diff() / data['开盘价'].shift()
            y = ret.iloc[2:].dropna(how='all')  # 数据对齐
        elif cal_way == 'close':
            x = data.shift().iloc[1:, :].dropna(how='all')  # 避免未来数据
            ret = data['收盘价'].diff() / data['收盘价'].shift()
            y = ret.iloc[1:].dropna(how='all')  # 数据对齐
        return x, y

    def transform(self, file_dir, cal_way='open', test_size=0.7, shuffle=False):
        x, y = self.data_fix(file_dir, cal_way=cal_way)
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)
        return X_train, X_test, y_train, y_test

class MATransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_sma):
        super(MATransformer, self).__init__()
        self.n_sma = n_sma

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        sma = X.rolling(self.n_sma).mean()
        return (sma - X).iloc[:, 0]


file_dir = r"./RB_data.csv"
# 数据清洗分割
X_train, X_test, y_train, y_test = PreTransformer().transform(file_dir, cal_way='open')


def rolling_mean(x, n):
    return x.rolling(n).mean()
def add(x, y):
    return x + y
def minus(x, y):
    return x - y
def rolling_rank(x, n):
    return x.rolling(n).rank()
def pct_change(x, n):
    return x / x.shift(n) - 1

init_function = ['rolling_mean', 'add', 'minus', 'rolling_rank', 'pct_change']

# for func in init_function:
#
#     rolling_mean(X_train, 1)
#     dynamic = str(func) + "(" + ")"
#     result = eval(dynamic.lstrip().rstrip("="))


# from sympy import *
#
# x1, x2 = symbols('x1, x2')  # 声明符号变量，否则会认为没有定义
# a = rolling_mean(x1,x2)
# print(a)


from sklearn_test.gp import node, flist, paramnode, constnode
import sklearn_test.gp as gp

def exampletree():
    # if arg[0] > 3:
    #   return arg[1] + 5
    # else:
    #   return arg[1] - 2
    return gp.node(
        gp.ifw, [
            gp.node(gp.gtw, [gp.paramnode(0), gp.constnode(3)]),
            gp.node(gp.addw, [gp.paramnode(1), gp.constnode(5)]),
            gp.node(gp.subw, [gp.paramnode(1), gp.constnode(2)])
        ]
    )



if __name__ == '__main__':
    rf = gp.getrankfunction(gp.buildhiddenset())
    final = gp.evolve(2, 500, rf, mutationrate=0.2, breedingrate=0.1, pexp=0.7, pnew=0.1)









