import talib as ta
import pandas as pd
from numba import jit, int32
def CMO_signal(close, cmo_length, m_length):
    # 检验一下
    signal = pd.Series([0]*len(close), index=close.index)
    cmo = ta.CMO(close, cmo_length)
    ma = ta.MA(cmo, m_length)
    return ma

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
    return upper, middle, lower

def rsi_signal(close, timeperiod, up, down):
    signal = pd.Series([0] * len(close), index=close.index)
    rsi = ta.RSI(close, timeperiod=timeperiod)
    return rsi

def macd_signal(close, fastperiod, slowperiod, signalperiod):
    signal = pd.Series([0] * len(close), index=close.index)
    dif, dea, macd = ta.MACD(close,
                             fastperiod=fastperiod,
                             slowperiod=slowperiod,
                             signalperiod=signalperiod)
    return dif, dea, macd


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

    return AMA - DIF

def two_ma_signal(close, short_p, long_p):
    signal = pd.Series([0] * len(close), index=close.index)
    ma_short = ta.MA(close, short_p)
    ma_long = ta.MA(close, long_p)

    return ma_short, ma_long


def moment(close, range):
    moment = close/close.shift(range) - 1
    return moment

def vol(close, range):
    ret = close.diff()/close.shift()
    vol = ret.rolling(range).std()
    return vol





