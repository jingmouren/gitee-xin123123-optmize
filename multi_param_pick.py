import pandas as pd
from Base import vector_backtest
from formula_package import *
import matplotlib.pyplot as plt
from data_simulate import DataSim

class Multi_param_backtest:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        try:
            data = pd.read_csv(file_dir, encoding='GBK')
        except:
            data = pd.read_csv(file_dir)
        self.data = data

    def f(self, x, t):
        # sharpe_mean = x['夏普比率'].mean()
        # sharpe_shred = max(0.2, sharpe_mean)
        # x = x[x['夏普比率'] > sharpe_shred]
        x = x.sort_values(by=t, ascending=False)
        range_n = int(len(x)/2)
        max_x_sharpe = -100
        max_list = pd.DataFrame()
        for i in range(range_n, len(x.index)):
            temp = x.iloc[i - range_n:i, :]['夏普比率']
            x_mean = temp.mean()
            x_std = temp.std()
            x_sharpe = x_mean / x_std
            if x_sharpe > max_x_sharpe:
                max_x_sharpe = x_sharpe
                max_list = x.iloc[i - range_n:i, :]
        if len(max_list) != 0:
            # max_list = max_list[max_list['夏普比率']>0]
            return max_list.sort_values(by='夏普比率').iloc[-1:, :]
        else:
            pass
    def param_pick(self):
        name = set(self.data.columns) - {'Unnamed: 0', '年化收益率', '季度胜率', '夏普比率', '卡玛比率'}
        all_df = pd.DataFrame()
        for t in name:
            rest_t = sorted(name - {t})
            df = self.data.groupby(by=rest_t).apply(self.f, t)
            all_df = all_df.append(df)
        all_df = all_df.drop_duplicates()
        all_df = all_df.sort_values(by='夏普比率').iloc[-5:, :]
        return all_df

    def portfolio_cal(self, start_date, end_date, file_dir, stragety_fuc, cal_way='open'):
        test1 = vector_backtest(start_date, end_date, file_dir, cal_way=cal_way)
        df = self.param_pick().drop(columns={'Unnamed: 0', '年化收益率', '季度胜率', '夏普比率', '卡玛比率'})
        temp_jz = pd.DataFrame()
        for row in range(len(df.index)):
            params = df.iloc[row, :].to_dict()
            print(params)
            signal = stragety_fuc(test1.data['Close'], **params)
            test1.add_stragety(signal=signal)
            test1.run()
            ret = test1.jz.diff() / test1.jz.shift()
            if len(temp_jz) == 0:
                temp_jz = ret
            else:
                temp_jz = pd.concat([temp_jz, ret], axis=1)
        temp_jz = temp_jz.mean(axis=1)
        jz = (1 + temp_jz).cumprod()
        return jz

if __name__ == "__main__":
    start_date = '20100101'
    end_date = '20200601'
    file_dir = r"./data/RB_data.csv"

    sum1 = DataSim(start_date, end_date, file_dir)
    sum1.relative_cal()
    sim_data = sum1.random_cal()
    sim_data.to_csv('./data/sim_RB.csv')
    a = pd.DataFrame(sim_data['Close'])


    ###### rsi ###########
    xx = Multi_param_backtest('./参数表/rsi.csv')
    jz1 = xx.portfolio_cal(start_date, end_date, file_dir, rsi_signal)
    pd.DataFrame(jz1).to_csv('./stragety_data/rsi_signal.csv')

    ###### CMO ###########
    xx = Multi_param_backtest('./参数表/cmo.csv')
    jz2 = xx.portfolio_cal(start_date, end_date, file_dir, CMO_signal)
    pd.DataFrame(jz2).to_csv('./stragety_data/CMO_signal.csv')


    ###### bolling ###########
    xx = Multi_param_backtest('./参数表/bolling.csv')
    jz = xx.portfolio_cal(start_date, end_date, file_dir, boll_signal)
    pd.DataFrame(jz).to_csv('./stragety_data/boll_signal.csv')

    ###### macd ###########
    xx = Multi_param_backtest('./参数表/macd.csv')
    jz3 = xx.portfolio_cal(start_date, end_date, file_dir, macd_signal)
    pd.DataFrame(jz).to_csv('./stragety_data/macd_signal.csv')


    # 作图
    cmb = pd.concat([jz1, jz2, jz3, jz, a], axis=1).loc[start_date:end_date, :].ffill().bfill()
    cmb.columns = ['rsi', 'cmo', 'boll', 'macd', 'init']
    cmb = cmb/cmb.iloc[0, :]
    cmb.plot()
    plt.show()

