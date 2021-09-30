import pandas as pd
import numpy as np
# 1分钟数据合成为n分钟数据

def resample(fix_df, n):
    '''k线合成'''
    Open = fix_df['Open'].shift(n-1)
    Close = fix_df['Close']
    High = fix_df['High'].rolling(n).max()
    Low = fix_df['Low'].rolling(n).min()
    Volume = fix_df['Volume'].rolling(n).sum()
    df = pd.concat([Open, High, Low, Close, Volume], axis=1).dropna()
    return df.iloc[::n, :]

def mul_freq_produce(main_file, symbol_name, freq_list):
    filename1 = symbol_name + '_1min.csv'
    df = pd.read_csv(main_file + filename1).set_index('trade_time')
    for i in freq_list:
        filename_n = symbol_name + '_' + str(i) + 'min.csv'
        df_resample = resample(df, i)
        df_resample.to_csv(main_file + filename_n)


if __name__ == "__main__":
    main_file = './行情数据库/螺纹/分钟/'
    symbol_name = 'RB'
    freq_list = [5, 15, 30, 60]  # 合成周期
    mul_freq_produce(main_file, symbol_name, freq_list)