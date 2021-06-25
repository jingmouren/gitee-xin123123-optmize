import talib as ta
import pandas as pd
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
    upper, middle, lower = ta.BBANDS(close,
                                  timeperiod=timeperiod,
                                  # number of non-biased standard deviations from the mean
                                  nbdevup=std,
                                  nbdevdn=std,
                                  # Moving average type: simple moving average here
                                  matype=0)
    signal[close > upper] = 1
    signal[close < lower] = -1
    return signal

