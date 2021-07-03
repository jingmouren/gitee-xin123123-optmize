"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn（暂时放弃不用）.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn（暂时放弃不用）.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers

import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata

__all__ = ['make_fitness']

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from scipy import stats
from tqdm import tqdm


class _Account:
    def __init__(self,
                 t,
                 cash=1000000,
                 fee_rate=0,
                 skid=0,
                 stop_loss=-2,
                 profit_taking=2,
                 stop_time=2.5):
        self.cash = cash
        self.fee_rate = fee_rate
        self.skid = skid
        self.stop_loss = stop_loss
        self.profit_taking = profit_taking
        self.stop_time = stop_time

        self.equity = self.cash
        self.unused_cash = self.cash
        self.multiple = 300
        self.margin_rate = 0.1
        self.lng_pos = 0
        self.sht_pos = 0
        self.margin = 0
        self.open_order_id = []
        self.balance = 0

        self.wallet_df = pd.DataFrame(columns=['equity', 'cash', 'unused_cash', 'lng_pos', 'sht_pos',
                                               'margin', 'time', 'price', 'open_order_id'])
        self.order_df = pd.DataFrame(columns=['open_time', 'close_time', 'open_price', 'close_price',
                                              'direction', 'amount', 'profit', 'margin',
                                              'fee', 'state', 'close_type'])
        self.open_order_df = self.order_df.copy()
        self.return_ls = []

        self.wallet_df.loc[0] = [self.equity, self.cash, self.unused_cash, self.lng_pos, self.sht_pos,
                                 self.margin, t, np.nan, self.open_order_id.copy()]

    def open_order(self, price, t, direction, amount, change=False):
        price = price * (1 + self.skid * direction)
        value = self.multiple * price * amount
        if change:
            fee = 0
        else:
            fee = value * self.fee_rate
        margin = value * self.margin_rate
        if self.unused_cash > fee + margin:  # 可以开仓
            _ = datetime.strftime(t, '%Y%m%d%H%M%S')
            if self.order_df.shape[0] > 0:
                order_id = _ + '%d' % (sum(i[:-1] == _ for i in self.order_df.index[-2:]) + 1)
            else:
                order_id = _ + '1'
            self.order_df.loc[order_id] = [t, np.nan, price, np.nan,
                                           direction, amount, 0, margin,
                                           fee, True, np.nan]
            self.cash -= fee
            self.margin += margin
            self.open_order_id.append(order_id)
            if direction == 1:
                self.lng_pos += amount
            else:
                self.sht_pos += amount
        else:  # 可用现金不足，无法开仓
            pass

    def close_order(self, price, t, order_id, close_type, change=False):
        open_price, direction, amount, margin, open_fee, state = self.order_df.loc[
            order_id, ['open_price', 'direction', 'amount', 'margin', 'fee', 'state']
        ].values.tolist()

        if state:  # 可以平仓
            price = price * (1 - self.skid * direction)
            value = self.multiple * price * amount
            if change:
                fee = 0
            else:
                fee = value * self.fee_rate
            profit = (price - open_price) * amount * direction * self.multiple
            self.order_df.loc[order_id, ['close_time', 'close_price', 'profit', 'fee',
                                         'state', 'close_type']] = \
                [t, price, profit, fee + open_fee, False, close_type]
            self.cash -= fee
            self.cash += profit
            self.margin -= margin
            self.open_order_id.remove(order_id)
            if direction == 1:
                self.lng_pos -= amount
            else:
                self.sht_pos -= amount
        else:  # 订单状态错误，无法平仓
            pass

    def update_wallet(self, bid, ask, t):  # 接收下一tick数据前执行
        self.balance = self.lng_pos - self.sht_pos
        self.open_order_df = self.order_df.loc[self.open_order_id]
        self.equity = self.cash - ((self.open_order_df['direction'] * self.open_order_df['amount'] *
                                    self.open_order_df['open_price']).sum() -
                                   bid * self.lng_pos + ask * self.sht_pos) * self.multiple
        self.unused_cash = self.cash - self.margin
        self.wallet_df.loc[len(self.wallet_df)] = [self.equity, self.cash, self.unused_cash, self.lng_pos, self.sht_pos,
                                                   self.margin, t, (bid + ask) / 2, self.open_order_id.copy()]

    def max_drawdown(self):
        i = int(np.argmax((np.maximum.accumulate(self.return_ls) - self.return_ls) /
                          np.maximum.accumulate(self.return_ls)))
        if i == 0:
            return 0
        j = int(np.argmax(self.return_ls[:i]))  # 开始位置

        return (self.return_ls[j] - self.return_ls[i]) / self.return_ls[j]

    def sharpe(self):
        mean = (np.mean(self.return_ls) + 1)
        std = np.std(self.return_ls)
        sharpe = mean / std * np.sqrt(252 * 4 * 60 * 60 * 2)
        return std, sharpe

class _Fitness:
    def __init__(self, function, greater_is_better, wrap=True):
        pass

    def close_stop(self, bid, ask, t):
        if self.account.balance > 0:
            profit = bid - self.account.open_order_df.open_price
        elif self.account.balance < 0:
            profit = ask - self.account.open_order_df.open_price
        else:
            return {}

        close_order_1 = profit[profit >= self.account.profit_taking].index.tolist()
        close_order_2 = profit[profit <= self.account.stop_loss].index.tolist()

        time_delay = t - timedelta(seconds=self.account.stop_time)
        close_order_3 = self.account.open_order_df.loc[
            self.account.open_order_df['open_time'] <= time_delay].index.tolist()

        return {
            **{order: '止盈' for order in close_order_1},
            **{order: '止损' for order in close_order_2},
            **{order: '超时' for order in set(close_order_3) - set(close_order_1) - set(close_order_2)}
        }

    def handle_tick(self, t):
        bid = self.bid[t]
        ask = self.ask[t]

        # 开平仓
        signal_rk = self.sig_rk[t]

        if self.account.balance == 0:
            if signal_rk >= 90:
                self.account.open_order(ask, t, 1, 1)  # 开多仓
            elif signal_rk <= 10:
                self.account.open_order(bid, t, -1, 1)  # 开空仓

        elif self.account.balance > 0:
            if signal_rk >= 90:
                close_order = self.close_stop(bid, ask, t)  # 判断是否止盈/止损/超时
                if close_order:
                    order = list(close_order.keys())[0]
                    self.account.close_order(
                        price=bid, t=t, order_id=order, close_type=close_order.get(order), change=True)
                    self.account.open_order(bid, t, 1, 1, change=True)  # 多换（无手续费）
                    for order in list(close_order.keys())[1:]:
                        self.account.close_order(
                            price=bid, t=t, order_id=order, close_type=close_order.get(order))
                else:
                    self.account.open_order(ask, t, 1, 1)  # 加多
        elif self.account.balance < 0:
            if signal_rk <= 10:
                close_order = self.close_stop(bid, ask, t)  # 判断是否止盈/止损/超时
                if close_order:
                    order = list(close_order.keys())[0]
                    self.account.close_order(
                        price=ask, t=t, order_id=order, close_type=close_order.get(order), change=True)
                    self.account.open_order(ask, t, -1, 1, change=True)  # 空换（无手续费）
                    for order in list(close_order.keys())[1:]:
                        self.account.close_order(
                            price=ask, t=t, order_id=order, close_type=close_order.get(order))
                else:
                    self.account.open_order(bid, t, -1, 1)  # 加空
        else:  # We never get here
            pass

        # 更新账户信息
        self.account.update_wallet(bid, ask, t)

    def handel_day(self, date):
        # 每天 9:50-11:30 、13:00-14:55 交易
        dttm_ls = [dttm for dttm in self.dttm_index if dttm.date() == date]
        dttm_order = [dttm for dttm in dttm_ls if time(9, 50) < dttm.time() < time(14, 55)]
        for dttm in tqdm(dttm_order):
            self.handle_tick(dttm)

        # 每天 14:55 强行平仓
        if self.account.open_order_id:
            t = np.min([dttm for dttm in dttm_ls if dttm.time() >= time(14, 55)])
            bid, ask = self.bid[t], self.ask[t]
            for order_id in self.account.open_order_id:
                if self.account.open_order_df.loc[order_id, 'direction'] == 1:
                    self.account.close_order(price=bid, t=t, order_id=order_id, close_type='收盘')
                else:
                    self.account.close_order(price=ask, t=t, order_id=order_id, close_type='收盘')
            self.account.update_wallet(bid, ask, t)

    def fit(self):
        if self.data['signal'].isna().sum() / len(self.data) > 0.001:
            return np.nan
        else:
            date_ls = list(set(dttm.date() for dttm in self.dttm_index))
            date_ls.sort()
            for date in date_ls:
                self.handel_day(date)




def make_fitness(function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score, weights=w)


weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss}
