import pandas as pd
df = pd.DataFrame()
for i in range(10):
    tt = pd.read_csv(str(i)+'_result_all.csv')
    if len(df) == 0:
        df = tt
    else:
        df = pd.concat([df, tt], axis=0)

df['name'] = df['timeperiod'].astype('str') + '-'+df['std'].astype('str')
df = df.groupby('name').mean().dropna()
df = df.sort_values(by='夏普比率')
print(df)