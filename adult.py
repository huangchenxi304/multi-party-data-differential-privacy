# 获取原始数据
import d_SVM
import drawing_v3
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
# 获取原始数据
data_raw = d_SVM.dataDigitize("data/adult_new.csv")
# 将特征fnlwgt离散为5个维度
data_raw['fnlwgt'] = pd.cut(data_raw['fnlwgt'], 5)
# data, labels, names = funcs.data_clean(data_raw)
adjustment_f = ['hours-per-week', 'education-num', 'fnlwgt', 'age']

data1 = data_raw[adjustment_f]
data2 = data_raw.drop(adjustment_f, axis=1)

best_f = list((data2.drop('lable', axis=1)).columns)
label = 'lable'


drawing_v3.run_pic(10, 'adult', data_raw, adjustment_f, best_f, label, 2)
