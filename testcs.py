import numpy as np
import pandas as pd
import funcs

# 读取数据
d1 = pd.read_csv('data/d1.csv')
d2 = pd.read_csv('data/d2.csv')
d = pd.read_csv('data/d_new.csv')

cs_1 = funcs.cs(d1)
print('对D1运行：\n', cs_1)
print('------------------')
cs_2 = funcs.cs(d2)
print('对D2运行：\n', cs_2)
print('------------------')
mcd = np.mean([cs_1['CS_mean'], cs_2['CS_mean']])
print('MCD为', mcd)
print('------------------')
cs_3 = funcs.cs(d)
print('对D运行，阈值为0.5：\n',cs_3)
print('------------------')
cs_3_mcd = funcs.cs(d, threshold=mcd)
print('对D运行，阈值为mcd：\n', cs_3_mcd)
print('------------------')

# 清洗数据
ad_x, ad_y, names = funcs.data_clean(d)
x = ad_x.values
y = list(ad_y)
# 特征选择
best_feature_set_names, adjusted_feature_set_names = funcs.randomized_lasso(x, y, names, best_features=2)
# 改名
best_feature_set_names, adjusted_feature_set_names = funcs.cast_features(best_feature_set_names,
                                                                         adjusted_feature_set_names)