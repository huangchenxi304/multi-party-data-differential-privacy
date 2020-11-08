import itertools
import algo_2_count
import d_SVM
import numpy as np
import funcs
import matplotlib.pyplot as plt
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


# 获取原始数据
data_raw = pd.read_csv("data/adult_new.csv")
data = d_SVM.dataDigitize("data/adult_new.csv")
data['fnlwgt'] = pd.cut(data['fnlwgt'],5)
# data, labels, names = funcs.data_clean(data_raw)
adjustment_features = ['hours-per-week', 'education-num', 'fnlwgt', 'age']
data1 = data[adjustment_features]
data2 = data.drop(adjustment_features ,axis=1)

data3 = data2.drop('lable',axis=1)
best_features = list(data3.columns)
print(best_features)
label = 'lable'




# 增加噪声
def add_laplace_noise(data_list, delta_f, privacy_budget):
    laplace_noise = np.random.laplace(0, delta_f / privacy_budget, len(data_list))  # 为原始数据添加噪声
    return laplace_noise + data_list


# print(add_laplace_noise(count_outlook,1,1))

# 根据调整特征系数对调整特征集的特征进行组合，生成备选方案
def partitionC(adjustment_features):
    # 两两组合，返回备选方案c
    c_raw  = [list(i) for i in itertools.combinations(adjustment_features, 2)]
    for x in c_raw:
        x.append(label)
    return c_raw





# 效用函数1
def gainInfo(feature_name):
    # 合并后数据集Dp的初始信息熵
    info_entropy_Dp = calInfoEntropy(data)
    # print(info_entropy_Dp)

    # 根据特征名称获取特征取值的count
    count_a_feature = data[feature_name].value_counts()

    # 根据特征取值划分子集
    subset = [data.loc[data[feature_name] == value] for value in count_a_feature._index]
    # for x in subset:
    #     print(x)

    # 计算各子集的信息熵
    subset_entropy = [calInfoEntropy(x) for x in subset]
    # print(subset_entropy)

    feature_chance = np.array([x / np.sum(count_a_feature) for x in count_a_feature])

    # 计算信息增益
    gain_info = info_entropy_Dp - np.matmul(feature_chance, subset_entropy)

    return gain_info


# 计算信息熵
def calInfoEntropy(data, label="lable"):
    # 计算标签取值count情况
    label_data = data[label].value_counts()
    # 计算标签取值概率
    chance = [x / np.sum(label_data) for x in label_data]
    # 计算信息熵
    info_entropy = -(np.sum(np.fromiter((x * np.log2(x) for x in chance), float)))
    return info_entropy


# 效用函数1
def utilityFunction1(feature_names):
    # 备选方案的信息增益为 b 个特征增益的和
    return np.sum(gainInfo(x) for x in feature_names)


# n方数据集平均相关度
def MCD():
    n_party_data = [data1, data2]

    return np.mean([funcs.cs(x)['CS_mean'] for x in n_party_data])


mcd = MCD()


# 效用函数2
def utilityFunction2(feature_names):

    cs_ci = funcs.cs(data[feature_names],mcd)['CS_i']


    return MCD() / cs_ci


# 使用效用函数1的指数机制1
def exponentialMechanism1(privacy_budget, feature_names, delta_u=1):

    return np.exp(privacy_budget * utilityFunction1(feature_names) / (2 * delta_u))


# 使用指数机制1的概率
def chanceExp1(privacy_budget, c, ci):
    exp_sum = np.sum(exponentialMechanism1(privacy_budget, x) for x in c)

    return exponentialMechanism1(privacy_budget, ci) / exp_sum


# 使用效用函数2的指数机制
def exponentialMechanism2(privacy_budget, feature_names, delta_u=1):

    return np.exp(privacy_budget * utilityFunction2(feature_names) / (2 * delta_u))


# 使用指数机制2的概率
def chanceExp2(privacy_budget, c, ci):
    exp_sum = np.sum(exponentialMechanism2(privacy_budget, x) for x in c)

    return exponentialMechanism2(privacy_budget, ci) / exp_sum


def selectCi1(privacy_budget, c, best_features):
    # 根据效用函数1选出的ci
    result1 = sorted(zip(map(lambda x: chanceExp1(privacy_budget, c, x), c), c), reverse=True)
    # 将选出的最佳ci加入best features
    return best_features + list(result1[0])[1]


def selectCi2(privacy_budget, c, best_features):
    # 根据效用函数2选出的ci
    result2 = sorted(zip(map(lambda x: chanceExp2(privacy_budget, c, x), c), c), reverse=True)

    # 将选出的最佳ci加入best features
    return best_features + list(result2[0])[1]




def MAE1(privacy_budget):
    result1 = selectCi1(1, partitionC(adjustment_features),best_features)


    u1 = algo_2_count.noise_count_error(data[result1], funcs.cs(data[result1],mcd)['CS_i'], privacy_budget)
    print(u1)
    return u1


def MAE2(privacy_budget):
    result2 = selectCi2(1, partitionC(adjustment_features),best_features)


    u2 = algo_2_count.noise_count_error(data[result2], funcs.cs(data[result2],mcd)['CS_i'], privacy_budget)

    print(u2)
    return u2


def painting1(n):
    x = []
    y = []
    for i in range(n):
        y.append(MAE1(i / n))
        x.append(i / n)
    return x, y


def painting2(n):
    x = []
    y = []
    for i in range(n):
        y.append(MAE2(i / n))
        x.append(i / n)
    fileObject = open('u1-y.txt', 'w')
    for ip in y:
        fileObject.write(ip)
        fileObject.write('\n')
    fileObject.close()
    fileObject2 = open('u1-x.txt', 'w')
    for ip in x:
        fileObject2.write(ip)
        fileObject2.write('\n')
    fileObject2.close()
    return x, y

if __name__ == '__main__':
    n = 50
    x1, y1 = painting1(n)
    # x2, y2 = painting2(n)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.xlabel('privacy_budget')
    plt.ylabel('MAE1')
    plt.plot(x1, y1)
    plt.savefig("n = " + str(n))
    plt.show()


