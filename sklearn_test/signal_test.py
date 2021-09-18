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


class Strategy(BaseEstimator):
    def __init__(self, type='mean', m=5):
        self.type = type
        self.m = m

    def fit(self, X_train, y_train):

        return self

    def predict(self, X_predict):
        signal = pd.Series(index=X_predict.index).fillna(0)

        if self.type == 'mean':
            diff = X_predict
            signal[diff > 0] = 1
            signal[diff < 0] = -1

        return signal

    def score(self, X, y):
        signal_df = self.predict(X)
        pnl_ratio = signal_df * y
        nav = (1 + pnl_ratio).cumprod()
        total_return = nav.iloc[-1] - 1
        return total_return


file_dir = r"./RB_1d.csv"
# 数据清洗分割
X_train, X_test, y_train, y_test = PreTransformer().transform(file_dir, cal_way='open')
X_train = X_train.iloc[:, 0]
X_test = X_test.iloc[:, 0]

# 单因子衍生，即指标计算，研究人员自行设置
ma6_factor_test = MATransformer(6).transform(X_train)
stragety1 = Strategy()

parameters = {'type': ('mean', ),
              'm': list(range(5,240,5))
              }

clf = GridSearchCV(stragety1, parameters)
clf.fit(ma6_factor_test, y_train)
print(clf)
print(clf.grid_scores_)





