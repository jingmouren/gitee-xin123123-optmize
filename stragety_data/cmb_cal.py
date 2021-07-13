import pandas as pd
import matplotlib.pyplot as plt


def multi_plot(file_name):
    df = pd.DataFrame()
    for f in file_name:
        temp = pd.read_csv(f)
        temp.columns = ['date', f.split('.')[0]]
        temp = temp.set_index('date')
        if len(df) == 0:
            df = temp
        else:
            df = pd.concat([df, temp], axis=1)
    ret = (df.diff()/df.shift()).dropna()
    jz = (1 + ret.mean(axis=1)).cumprod()

    jz.plot()
    plt.show()

if __name__ == "__main__":
    file_name = ['boll_signal.csv', 'rsi_signal.csv', 'CMO_signal.csv',
                 'macd_signal.csv', 'Aberration_signal.csv', 'DMA_signal.csv', 'two_ma_signal.csv']
    multi_plot(file_name)