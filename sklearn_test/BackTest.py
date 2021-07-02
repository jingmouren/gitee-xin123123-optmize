from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def data_get(filename):
    with open(filename, 'rb') as f:
        df = pd.read_csv(f, encoding='gbk')
    df = df.set_index(["symbol", "trade_date"])
    df = df.to_xarray().to_array().T
    y = df.sel(variable='close')
    trade_status = df.sel(variable='trade_status')
    df = df.drop(['close'], dim='variable')
    df = df.drop(['trade_status'], dim='variable')
    return df, y, trade_status
    pass


class MATransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_sma):
        super(MATransformer, self).__init__()
        self.n_sma = n_sma

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def transform(self, X):
        return X

class XXTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        super(XXTransformer, self).__init__()

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        return self

    def transform(self, X):
        return X

class YYTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, param):
        super(YYTransformer, self).__init__()
        self.param = param
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = xr.DataArray(X.reshape(self.param.shape), dims=self.param.dims, coords=self.param.coords)
        return X

class Strategy(BaseEstimator):
    def __int__(self):
        """初始化kNN分类器"""
        self.k = 1
        """变量前加_，表示该变量为类私有，其它类不能随便操作"""
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        """
        为了和scikit-learn库的规则一样，此处一般返回模型本身，
        可使封装好的算法与scikit-learn中其它方法更好结合
        """
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据，返回x的预测结果"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = np.Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        """kNN算法的显示名称"""
        return "KNN(k = %d)" % self.k

# 新建无量纲化对象
#新建计算缺失值的对象
with open('df.csv', 'rb') as f:
    df = pd.read_csv(f, encoding='gbk')
del df['Unnamed: 0']
df['trade_status'] = df['trade_status'].apply(lambda x: 1 if x == "正常交易" else 0)

df = pd.pivot(df, columns='symbol', index='trade_date').sort_index()  # 展开
df.columns = df.columns.remove_unused_levels()  # 使得对象的MultiIndex对象和其实际显示出来的索引保持一致了
close = df.close  # 价格数据
# 数据分割
X_train, X_test, y_train, y_test = train_test_split(df, close, test_size=0.2, shuffle=False)
y_train = y_train.shift().iloc[1:, :].dropna(how='all')  # 避免未来数据
y_test = y_test.shift().iloc[1:, :].dropna(how='all')  # 避免未来数据
X_train = X_train.iloc[1:, :].dropna(how='all')  # 数据对齐
X_test = X_test.iloc[1:, :].dropna(how='all')  # 数据对齐



class PreTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        super(PreTransformer, self).__init__()

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.ffill()  # 前值填充
        X = X.stack().reset_index()
      #  X = X.groupby('symbol').apply(lambda x: x.ffill())  # 前值填充
        X = X.groupby('trade_date').apply(lambda x: x.fillna(x.mean()))  # 均值填充
        X = pd.pivot(X, columns='symbol', index='trade_date')
        X.columns = X.columns.remove_unused_levels()
        train_dict = {}
        for factor in X.columns.levels[0]:
            train_dict[factor] = X[factor]
        return train_dict

class MATransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_sma):
        super(MATransformer, self).__init__()
        self.n_sma = n_sma

    def fit(self, X):
        return self
    def ma(self,x):
        return x.rolling(self.n_sma).mean()

    def transform(self, X):
        new_X = {}
        for f_name, factor in X.items():
            ma_name = f_name + '_' + str(self.n_sma) + "ma"
            new_X[ma_name] = self.ma(factor)
        X.update(new_X)
        return X

class IcFactorCmb(TransformerMixin, BaseEstimator):
    def __init__(self, n_len):
        super(IcFactorCmb, self).__init__()
        self.n_len = n_len

    def fit(self, X, y=None):
        return self
    def factor_ret_cal(self, x, y):
        '''
        因子收益率计算
        '''
        ret = (y.diff() / y.shift()).fillna(0)

        ret_df = pd.DataFrame()
        for key, value in x.items():
            name = set(ret.columns) & set(value.columns)
            fac_ret = []
            for date in value.index:
                x = np.mat(value.loc[date, name].fillna(0)).T  # 有缺失值，需要处理
                r = np.mat(ret.loc[date, name].fillna(0)).T  # 有缺失值，需要处理
                daily_ret = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(r)[0,0]
                fac_ret.append(daily_ret)
            fac_ret = pd.DataFrame(fac_ret, index=value.index, columns=[key])
            if len(ret_df) == 0:
                ret_df = fac_ret
            else:
                ret_df = pd.concat([ret_df, fac_ret], axis=1)
        return ret_df
        pass
    def IC_cal(self, x, y):
        '''
        因子收益率计算
        '''
        ret = (y.diff() / y.shift()).fillna(0)

        rankIC_df = pd.DataFrame()
        for key, value in x.items():
            name = set(ret.columns) & set(value.columns)
            rankIC = []
            for date in value.index:
                x = value.loc[date, name].fillna(value.loc[date, name].mean())  # 有缺失值，需要处理
                r = ret.loc[date, name].fillna(ret.loc[date, name].mean())  # 有缺失值，需要处理
                x = x.sort_values()
                x = pd.Series(range(len(x)), index=x.index)
                r = r.sort_values()
                r = pd.Series(range(len(r)), index=r.index)
                temp = pd.concat([x, r], axis=1, sort=True)
                dailyIC = temp.corr().iloc[1, 0]
                rankIC.append(dailyIC)
            rankIC = pd.DataFrame(rankIC, index=value.index, columns=[key])
            if len(rankIC_df) == 0:
                rankIC_df = rankIC
            else:
                rankIC_df = pd.concat([rankIC_df, rankIC], axis=1)

        return rankIC_df

    def transform(self, X, y=None):
        rankIC = self.IC_cal(X, y)
        rankIC = rankIC.div(rankIC.sum(axis=1), axis=0)  # 归一
        # rankIC加权
        fac_cmb = pd.DataFrame()
        for key, value in X.items():
            temp = value.mul(rankIC[key], axis=0).fillna(0)
            if len(fac_cmb) == 0:
                fac_cmb = temp
            else:
                fac_cmb += temp
        fac_cmb = fac_cmb / len(X.keys())

        return fac_cmb

class Analysis:
    def __init__(self):
        pass

    def yearlyRet(self, series):
        '''
        传入净值数据
        返回年化收益率
        '''
        range_ret = series[-1] / series[0] - 1  # 期间收益率
        range_n = len(series) / 244  # 投资年份长度
        yearlyRet = (range_ret + 1) ** (1 / range_n) - 1  # 复利年化收益率
        return yearlyRet

    def QuarterlyRet(self, df):
        '''
        传入净值数据
        返回季度收益率数据
        '''
        # 每个季度末的净值数据
        s_df_q = df.resample('1Q', label='right', closed='right').last()
        # 拼接第一个净值数据
        s_first = pd.Series([df.iloc[0]], index=[df.index[0]])
        s_df_q = pd.concat([s_first, s_df_q], axis=0)
        ret_q = (s_df_q.diff() / s_df_q.shift()).dropna()
        return ret_q

    def WinRatio(self, ret1):
        '''
        传入一组对比收益率
        返回胜率
        '''
        ret1 = ret1.apply(lambda x: 1 if x > 0 else 0)
        win_ratio = ret1.sum() / len(ret1)
        return win_ratio

    def yearlyVol(self, df):
        '''
        传入日频净值数据
        返回年化波动率
        '''
        ret = (df.diff() / df.shift()).fillna(0)
        vol = ret.std() * (244 ** 0.5)
        return vol

    def SharpeRatio(self, df):
        riskFreeRate = 0.04  # 可接受的最小收益：无风险收益
        # 年化收益率
        yearly_ret = self.yearlyRet(df)
        # 年化波动率
        yearly_vol = self.yearlyVol(df)
        sharpe_ratio = (yearly_ret - riskFreeRate) / yearly_vol
        return sharpe_ratio

    def MaxTraceBack(self, df):
        '''
        返回最大连续回撤
        '''
        trace_list = []
        net_value = np.array(df)
        for num, value in enumerate(net_value):
            if num == 0:
                trace_list.append(0)
            else:
                trace = 1 - value / max(net_value[:num + 1])
                trace_list.append(trace)

        traceBack = trace_list
        max_traceback = max(traceBack)
        return max_traceback

    def calma(self, df):
        '''
        返回卡玛比率
        '''
        yearly_ret = self.yearlyRet(df)
        max_traceback = self.MaxTraceBack(df)
        calma_ratio = yearly_ret / max_traceback
        return calma_ratio

    def analysis(self, s_df):
        '''
        绩效分析
        '''
        # 年化收益率
        yearly_ret = self.yearlyRet(s_df)
        # 获取季度收益率
        ret_Q = self.QuarterlyRet(s_df)
        # 获取季度胜率
        win_ratio = self.WinRatio(ret_Q)
        # 年化波动率
        yearly_vol = self.yearlyVol(s_df)
        # 夏普比率
        sharpe_ratio = self.SharpeRatio(s_df)
        # 最大连续回撤
        max_traceback = self.MaxTraceBack(s_df)
        # 卡玛比率
        calma_ratio = self.calma(s_df)

        # 拼接
        result = pd.DataFrame([yearly_ret, win_ratio, sharpe_ratio, calma_ratio]
                              , index=["年化收益率", "季度胜率", "夏普比率", "卡玛比率"]
                              , columns=["绩效数据"])
        return result.T
class Portfolio(TransformerMixin, BaseEstimator, Analysis):
    def __init__(self, n_len=18):
        self.n_len = n_len  # 回归需要的数据长度
        pass

    def fit(self, X, y):
        ret = (y.diff() / y.shift()).fillna(0)
        value = X.iloc[-self.n_len:, :]  # 截取数据长度
        name = set(ret.columns) & set(value.columns)
        fac_ret = []
        for date in value.index:  # 计算每天的因子收益率
            x = np.mat(value.loc[date, name].fillna(0)).T  # 有缺失值，需要处理
            r = np.mat(ret.loc[date, name].fillna(0)).T  # 有缺失值，需要处理
            daily_ret = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(r)[0, 0]
            fac_ret.append(daily_ret)
        fac_ret = pd.Series(fac_ret, index=value.index)
        self.next_ret = fac_ret.ewm(halflife=90, ignore_na=True, adjust=True).mean().iloc[-1]  # 以均值作为未来一期收益率的预测
        # self.next_ret = fac_ret.mean()
        return self
    def Weight_cal(self, X):
        def score_cal(x):
            x[x < 0] = 0
            x = x.sort_values()
            return (x[-50:] / x[-50:].sum())
        return X.apply(score_cal, axis=1)
        pass
    def predict(self, X):
        y_hat = self.next_ret * X  # 用历史数据拟合出来的预测值y
        weight = self.Weight_cal(y_hat).fillna(0)
        return weight
        pass
    def score(self, X, y):
        weight = self.predict(X)
        ret = (y.diff() / y.shift()).fillna(0)
        portfolio_ret = (ret * weight).fillna(0).sum(axis=1)
        jz = (portfolio_ret + 1).cumprod()
        jz.index = pd.to_datetime(jz.index, format="%Y%m%d")
        # sharpe_ratio = self.SharpeRatio(jz)
        calma_ratio = self.calma(jz)
        return calma_ratio

        pass

# 数据预处理，缺失值填充
X_train = PreTransformer().fit(X_train).transform(X_train)

# 单因子衍生，即指标计算，研究人员自行设置，要确保返回dict形式
ma6_factor = MATransformer(6).fit(X_train).transform(X_train)
# 多因子组合
F_factor = IcFactorCmb(3).fit(ma6_factor, y_train).transform(ma6_factor, y_train)
# 向量化计算每日收益率
P = Portfolio().fit(F_factor, y_train).predict(F_factor)
print(P)
# 暂时略过，后期考虑pipeline
# 预测
# 数据预处理，缺失值填充
X_test = PreTransformer().fit(X_test).transform(X_test)

# 单因子衍生，即指标计算，研究人员自行设置，要确保返回dict形式
ma6_factor_test = MATransformer(6).fit(X_test).transform(X_test)
# 多因子组合
F_factor_test = IcFactorCmb(3).fit(ma6_factor_test, y_test).transform(ma6_factor_test, y_test)
# 计算分数
P_test = Portfolio().fit(F_factor, y_train).score(F_factor_test, y_test)



print(P_test)

# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# x = np.arange(9).reshape(3,3)
# print(x)
# poly = PolynomialFeatures(2)
# poly.fit_transform(x)












'''

step1 = ('Imputer', SimpleImputer())
# step1 = ('XXTransformer', XXTransformer())
step2 = ('MinMaxScaler', StandardScaler())
# 自定义因子生成方式
param = X_train
step3 = ('YYTransformer', YYTransformer(param))
step31 = ('MATransformer', MATransformer(10))
from sklearn.linear_model import LogisticRegression
step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
#新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
step4 = ('Strategy', Strategy)
#新建流水线处理对象
#参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
# pipeline = Pipeline(steps=[step1, step31, step4])


# for dd in X_train['variable']:
#     pipeline.fit(X_train.sel(variable=dd))
# y_pred = pipeline.predict(X_test)
# print(y_pred)
pipeline = Pipeline(steps=[step1, step2, step4])
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse
import numpy as np

#部分并行处理，继承FeatureUnion
class FeatureUnionExt(FeatureUnion):
    #相比FeatureUnion，多了idx_list参数，其表示每个并行工作需要读取的特征矩阵的列
    def __init__(self, transformer_list, idx_list, n_jobs=1, transformer_weights=None):
        self.idx_list = idx_list
        FeatureUnion.__init__(self, transformer_list=map(lambda trans:(trans[0], trans[1]), transformer_list), n_jobs=n_jobs, transformer_weights=transformer_weights)

    #由于只部分读取特征矩阵，方法fit需要重构
    def fit(self, X, y=None):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        transformers = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit方法
            delayed(_fit_one_transformer)(trans, X[:,idx], y)
            for name, trans, idx in transformer_idx_list)
        self._update_transformer_list(transformers)
        return self

    #由于只部分读取特征矩阵，方法fit_transform需要重构
    def fit_transform(self, X, y=None, **fit_params):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        result = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit_transform方法
            delayed(_fit_transform_one)(trans, name, X[:,idx], y,
                                        self.transformer_weights, **fit_params)
            for name, trans, idx in transformer_idx_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    #由于只部分读取特征矩阵，方法transform需要重构
    def transform(self, X):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        Xs = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入transform方法
            delayed(_transform_one)(trans, name, X[:,idx], self.transformer_weights)
            for name, trans, idx in transformer_idx_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs


step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))


# 处理计算持仓信息

# 计算每个bar的收益率

# 计算组合收益率

'''





