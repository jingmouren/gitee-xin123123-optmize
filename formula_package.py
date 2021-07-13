import talib as ta
import pandas as pd
from numba import jit, int32
def CMO_signal(close, cmo_length, m_length):
    # 检验一下
    signal = pd.Series([0]*len(close), index=close.index)
    cmo = ta.CMO(close, cmo_length)
    ma = ta.MA(cmo, m_length)
    signal[cmo > ma] = 1
    signal[cmo < ma] = -1
    return signal

def boll_signal(close, timeperiod, std):
    signal = pd.Series([0] * len(close), index=close.index)
    close = close.rolling(10).mean()
    upper, middle, lower = ta.BBANDS(close,
                                  timeperiod=timeperiod,
                                  # number of non-biased standard deviations from the mean
                                  nbdevup=std,
                                  nbdevdn=std,
                                  # Moving average type: simple moving average here
                                  matype=0)
    signal[close > upper] = 1
    signal[close < lower] = -1

    # vol = close.diff() / close.shift()
    # vol = vol.rolling(30).std()
    # aa = []
    # for i in range(len(vol)):
    #     if i < 120:
    #         aa.append(1)
    #         continue
    #     tt = vol.iloc[:i].rank(pct=True)
    #     aa.append(tt.iloc[-1])
    # aa = pd.Series(aa, index=close.index)
    # signal[aa < 0.3] = 0

    return signal

def rsi_signal(close, timeperiod, up, down):
    signal = pd.Series([0] * len(close), index=close.index)
    rsi = ta.RSI(close, timeperiod=timeperiod)

    signal[rsi > up] = 1
    signal[rsi < down] = -1
    return signal

def macd_signal(close, fastperiod, slowperiod, signalperiod):
    signal = pd.Series([0] * len(close), index=close.index)
    dif, dea, macd = ta.MACD(close,
                             fastperiod=fastperiod,
                             slowperiod=slowperiod,
                             signalperiod=signalperiod)


    signal[(dif > 0) & (dea > 0) & (macd > 0)] = 1
    signal[(dif < 0) & (dea < 0) & (macd < 0)] = -1
    return signal

def Aberration_signal(close, timeperiod, std):
    signal = pd.Series([0] * len(close), index=close.index)
    upper, middle, lower = ta.BBANDS(close,
                                  timeperiod=timeperiod,
                                  # number of non-biased standard deviations from the mean
                                  nbdevup=std,
                                  nbdevdn=std,
                                  # Moving average type: simple moving average here
                                  matype=0)
    s = []
    for i in range(len(signal)):
        if i <= timeperiod:
            s.append(0)
            continue
        if close[i] > upper[i]:
            s.append(1)
        elif close[i] < lower[i]:
            s.append(-1)
        elif s[-1] == 1 and close[i] < middle[i] and close[i-1] >= middle[i-1]:
            s.append(0)
        elif s[-1] == -1 and close[i] > middle[i] and close[i-1] <= middle[i-1]:
            s.append(0)
        else:
            s.append(s[-1])
    signal = pd.Series(s, index=close.index)
    return signal

def DMA_signal(close, N1, N2, M):
    ''' 计算DMA
        输入参数：
            symbol <- str  标的代码
            start_time <- str  开始时间
            end_time <- 结束时间
            N1 <- 大周期均值
            N2 <- 小周期均值
        输出参数：
            DMA <- dataframe
    '''
    signal = pd.Series([0] * len(close), index=close.index)
    MA1 = close.rolling(N1).mean()
    MA2 = close.rolling(N2).mean()
    DIF = MA1 - MA2
    AMA = DIF.rolling(M).mean()
    signal[DIF > AMA] = 1
    signal[DIF < AMA] = -1

    return signal

def two_ma_signal(close, short_p, long_p):
    signal = pd.Series([0] * len(close), index=close.index)
    ma_short = ta.MA(close, short_p)
    ma_long = ta.MA(close, long_p)
    signal[ma_short > ma_long] = 1
    signal[ma_short < ma_long] = -1
    # vol = close.diff()/close.shift()
    # vol = vol.rolling(60).std()
    # aa = []
    # for i in range(len(vol)):
    #     if i < 120:
    #         aa.append(1)
    #         continue
    #     tt = vol.iloc[:i].rank(pct=True)
    #     aa.append(tt.iloc[-1])
    # aa = pd.Series(aa, index=close.index)
    # signal[aa < 0.5] = 0

    return signal




