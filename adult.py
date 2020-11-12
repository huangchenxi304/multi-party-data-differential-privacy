# 获取原始数据
import d_SVM
import itertools
import algo_2_count
import numpy as np
import funcs
import warnings
import pandas as pd
import algorithm2_v3

warnings.filterwarnings("ignore")
data_titanic = d_SVM.dataDigitize("data/titanic.csv")
data_raw = d_SVM.dataDigitize("data/adult_new.csv")
# 将特征fnlwgt离散为5个维度
data_raw['fnlwgt'] = pd.cut(data_raw['fnlwgt'], 5)
# data, labels, names = funcs.data_clean(data_raw)
adjustment_f = ['hours-per-week', 'education-num', 'fnlwgt', 'age']

data1 = data_raw[adjustment_f]
data2 = data_raw.drop(adjustment_f, axis=1)

best_f = list((data2.drop('lable', axis=1)).columns)
label = 'lable'


algorithm2_v3.mae1(data_raw, 1, best_f, adjustment_f, label)
algorithm2_v3.mae2(data_raw, 1, best_f, adjustment_f, label)
