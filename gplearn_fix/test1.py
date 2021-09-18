
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from gplearn_fix.genetic import SymbolicTransformer, BaseSymbolic
start_date = '20100101'
end_date = '20200601'
file_dir = r"../data/RB_1d.csv"

data = pd.read_csv(file_dir)
data = data.set_index(data.columns[0])
data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
data.head()

features = (data.shift()).iloc[1:, :].ffill()
TARGET = (data.diff()/data.shift()).iloc[1:, 0]
TARGET.name = 'TARGET'
TARGET.head()

x_train, x_test, y_train, y_test = train_test_split(features, TARGET,
                                                        test_size=0.3, shuffle=True)
train_idx = x_train.index
test_idx = x_test.index



init_function = ['add', 'sub', 'mul', 'div','sqrt', 'log','inv','sin','max','min']
function_set = init_function



est_gp = SymbolicTransformer(generations=3,   # 公式进化的世代数量
                             population_size=1000, # 每一代生成因子数量
                             hall_of_fame=100,  # 备选因子的数量
                             n_components=10, # 最终筛选出的最优因子的数量
                             function_set=function_set, # 函数集
                             parsimony_coefficient=0.0005,  # 节俭系数
                             max_samples=0.9,  # 最大采样比例
                             verbose=1,
                             d_ls=[20,30,40])
# x_train = np.nan_to_num(x_train)
# y_train = np.nan_to_num(y_train)

est_gp.fit(x_train, y_train)
best_programs = est_gp._best_programs
best_programs_dict = {}

for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_,
                                       'length': p.length_}

best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness')
print(best_programs_dict)
