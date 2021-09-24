import pandas as pd
import numpy as np

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
        if yearly_vol == 0:
            return 0
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
        if max_traceback == 0:
            return 0
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
