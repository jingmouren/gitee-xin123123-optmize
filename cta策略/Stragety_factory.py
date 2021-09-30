import pandas as pd
from cta策略.Base.backtest import SimpleBacktest
import talib as ta
import numpy as np


class Aberration(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol, multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol, multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.n, self.p = n
    def signal_cal(self):
        n = self.n
        p = self.p

        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return
        upper, middle, lower = ta.BBANDS(
            self.his_data['close'],
            timeperiod=n,
            # number of non-biased standard deviations from the mean
            nbdevup=p,
            nbdevdn=p,
            # Moving average type: simple moving average here
            matype=0)
        close = self.his_data['close'][-1]
        signal = 0
        if close > upper[-1]:
            signal = 1
        elif close < lower[-1]:
            signal = -1
        elif close < middle[-1] and close > lower[-1] and self.last_hands > 0:
            signal = 0
        elif close > middle[-1] and close < upper[-1] and self.last_hands < 0:
            signal = 0
        elif self.last_hands > 0:
            signal = 1
        elif self.last_hands < 0:
            signal = -1

        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class BBI(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.n1, self.n2, self.n3, self.n4 = n
    def signal_cal(self):
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        n4 = self.n4

        if len(self.his_data['close']) < max(n1, n2, n3, n4):
            self.target_position(0, self.his_data['last'])
            return
        ma1 = np.mean(self.his_data['close'][-n1:])
        ma2 = np.mean(self.his_data['close'][-n2:])
        ma3 = np.mean(self.his_data['close'][-n3:])
        ma4 = np.mean(self.his_data['close'][-n4:])
        bbi = (ma1 + ma2 + ma3 + ma4) / 4
        close = self.his_data['close'][-1]

        signal = 0
        if close > bbi:
            signal = 1
        if close < bbi:
            signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass


class BIAS(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.n, self.p = n
    def signal_cal(self):
        n = self.n
        thred = self.p

        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return
        n_mean = np.mean(self.his_data['close'][-n:])

        close = self.his_data['close'][-1]
        bias = (close - n_mean) / n_mean * 100

        signal = 0
        if bias > thred:
            signal = 1
        elif close < -thred:
            signal = -1
        # elif self.last_hands > 0:
        #     signal = 1
        # elif self.last_hands < 0:
        #     signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class BOLL(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.n, self.p = n
    def signal_cal(self):
        n = self.n
        p = self.p

        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return
        upper, middle, lower = ta.BBANDS(
            self.his_data['close'],
            timeperiod=n,
            # number of non-biased standard deviations from the mean
            nbdevup=p,
            nbdevdn=p,
            # Moving average type: simple moving average here
            matype=0)
        close = self.his_data['close'][-1]
        signal = 0
        if close > upper[-1]:
            signal = 1
        elif close < lower[-1]:
            signal = -1
        elif self.last_hands > 0:
            signal = 1
        elif self.last_hands < 0:
            signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class CCI(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)

    def param_add(self, n):
        # 参数设置
        self.n = n
    def signal_cal(self):
        n = self.n
        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return

        cci = ta.CCI(self.his_data['high'], self.his_data['low'], self.his_data['close'], timeperiod=n)[-1]
        signal = 0
        if cci > 100:
            signal = 1
        elif cci < -100:
            signal = -1
        elif self.last_hands > 0:
            signal = 1
        elif self.last_hands < 0:
            signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class CMO(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)

    def param_add(self, n):
        # 参数设置
        self.n = n
    def signal_cal(self):
        n = self.n
        if len(self.his_data['close']) < n+1:
            self.target_position(0, self.his_data['last'])
            return

        cmo = ta.CMO(self.his_data['close'], timeperiod=n)[-1]
        if cmo > 0:
            signal = 1
        if cmo < 0:
            signal = -1
        if cmo == 0:
            signal = 0
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class DMA(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
        self.dma = []
        self.ama = []
    def param_add(self, n):
        # 参数设置
        self.n1, self.n2 = n

    def rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def signal_cal(self):
        n1 = self.n1
        n2 = self.n2

        if len(self.his_data['close']) < n2+n1:
            self.target_position(0, self.his_data['last'])
            return

        sma = np.mean(self.rolling_window(self.his_data['close'], n1), -1)
        lma = np.mean(self.rolling_window(self.his_data['close'], n2), -1)
        dma1 = sma[-1] - lma[-1]
        dma2 = sma[-2] - lma[-2]

        ama1 = np.mean(sma[-n1:] - lma[-n1:])
        ama2 = np.mean(sma[-n1 - 1:-1] - lma[-n1 - 1:-1])
        signal = 0
        if dma1 > 0 and dma1 - dma2 > 0 and ama1 > 0 and ama1 - ama2 > 0:
            signal = 1
        elif dma1 < 0 and dma1 - dma2 < 0 and ama1 < 0 and ama1 - ama2 < 0:
            signal = -1
        elif self.last_hands > 0:
            signal = 1
        elif self.last_hands < 0:
            signal = -1

        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class KDJ(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.short, self.long, self.mmid = n
    def signal_cal(self):
        n1 = self.short
        n2 = self.long
        n3 = self.mmid

        if len(self.his_data['close']) < n1 + n2+n3:
            self.target_position(0, self.his_data['last'])
            return
        slowk, slowd = ta.STOCH(self.his_data['high'],
                                self.his_data['low'],
                                self.his_data['close'],
                                fastk_period=n1,
                                slowk_period=n2,
                                slowk_matype=0,
                                slowd_period=n3,
                                slowd_matype=0)
        K = slowk[-1]
        D = slowd[-1]
        J = (3 * K) - (2 * D)
        signal = 0
        if K > D:
            signal = 1
        if K < D:
            signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class MA(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)

    def param_add(self, n):
        # 参数设置
        self.n = n
    def signal_cal(self):
        n = self.n
        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return
        signal = np.sign(self.his_data['close'][-1] - np.mean(self.his_data['close'][-n:]))
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class MACD(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.short, self.long, self.mmid = n
    def signal_cal(self):
        short = self.short
        long = self.long
        mmid = self.mmid

        if len(self.his_data['close']) < long + mmid:
            self.target_position(0, self.his_data['last'])
            return
        macd, signal, hist = ta.MACD(self.his_data['close'], fastperiod=short, slowperiod=long, signalperiod=mmid)
        signal = 0
        if macd[-1] > 0:
            signal = 1
        if macd[-1] < 0:
            signal = -1
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass


class ROC(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)

    def param_add(self, n):
        # 参数设置
        self.n = n
    def signal_cal(self):
        n = self.n
        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return
        signal = np.sign(self.his_data['close'][-1] / self.his_data['close'][-n] - 1)
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class RSI(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)

    def param_add(self, n):
        # 参数设置
        self.n = n
    def signal_cal(self):
        n = self.n
        if len(self.his_data['close']) < n:
            self.target_position(0, self.his_data['last'])
            return

        cci = ta.RSI(self.his_data['close'], timeperiod=n)[-1]
        signal = 0
        if cci > 90:
            signal = -1
        if cci < 10:
            signal = 1
        elif self.last_hands > 0 and cci < 30:
            signal = 1
        elif self.last_hands < 0 and cci > 70:
            signal = -1

        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class SMA(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)

    def param_add(self, n):
        # 参数设置
        self.n1, self.n2 = n
    def signal_cal(self):
        n1 = self.n1
        n2 = self.n2

        if len(self.his_data['close']) < n2:
            self.target_position(0, self.his_data['last'])
            return
        sma = np.mean(self.his_data['close'][-n1:])
        lma = np.mean(self.his_data['close'][-n2:])
        signal = np.sign(sma - lma)
        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass

class TRIX(SimpleBacktest):
    def __init__(self, start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                 multip, freq='1d', cal_way='open'):
        super().__init__(start_date, end_date, comission_rate, slip_point, min_point, init_cash, main_file, symbol,
                         multip, freq, cal_way)
    def param_add(self, n):
        # 参数设置
        self.n1, self.n2 = n
    def signal_cal(self):
        n1 = self.n1
        n2 = self.n2

        if len(self.his_data['close']) < n1*3 + n2:
            self.target_position(0, self.his_data['last'])
            return
        trix = ta.TRIX(self.his_data['close'], timeperiod=n1)[-n2:]
        matrix = np.mean(trix)

        signal = 0
        if trix[-1] > matrix:
            signal = 1
        if trix[-1] < matrix:
            signal = -1

        self.last_signal.append([signal, self.his_data['time']])
        hands = self.capital / self.multip / self.his_data['last'] * signal
        self.target_position(hands, self.his_data['last'])
        pass







