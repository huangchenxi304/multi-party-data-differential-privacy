import d_SVM
import numpy as np
import funcs

# 获取原始数据
data = d_SVM.dataDigitize("D:\学习\差分隐私\d.xlsx")
data1 = d_SVM.dataDigitize("D:\学习\差分隐私\d1.xlsx")
data2 = d_SVM.dataDigitize("D:\学习\差分隐私\d2.xlsx")

# 增加噪声
def add_laplace_noise(data_list, delta_f, privacy_budget):
    laplace_noise = np.random.laplace(0, delta_f/privacy_budget, len(data_list)) # 为原始数据添加噪声
    return laplace_noise + data_list

# print(add_laplace_noise(count_outlook,1,1))

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
    gain_info = info_entropy_Dp - np.matmul(feature_chance,subset_entropy)

    return gain_info

# 计算信息熵
def calInfoEntropy(data,label = "Play?"):
    # 计算标签取值count情况
    label_data = data[label].value_counts()
    # 计算标签取值概率
    chance = [x / np.sum(label_data) for x in label_data]
    # 计算信息熵
    info_entropy = -(np.sum(np.fromiter((x*np.log2(x) for x in chance),float)))
    return info_entropy


# 效用函数1
def utilityFunction1(feature_names):

    # 备选方案的信息增益为 b 个特征增益的和
    return np.sum(gainInfo(x) for x in feature_names)


# n方数据集平均相关度
def MCD():
    n_party_data = [data1,data2]



    return np.mean([funcs.cs(x)['CS_mean'] for x in n_party_data])


# 效用函数2
def utilityFunction2(feature_names):

    cs_ci = funcs.cs(data[feature_names])['CS_i']

    return MCD() / cs_ci


# 使用效用函数1的指数机制1
def exponentialMechanism1(privacy_budget, feature_names, delta_u = 1):

    return np.exp(privacy_budget * utilityFunction1(feature_names) / (2 * delta_u))


# 使用指数机制1的概率
def chanceExp1(privacy_budget, c, ci):

    exp_sum = np.sum(exponentialMechanism1(privacy_budget,x) for x in c)

    return exponentialMechanism1(privacy_budget,ci) / exp_sum


# 使用效用函数2的指数机制2
def exponentialMechanism2(privacy_budget, feature_names, delta_u = 1):

    return np.exp(privacy_budget * utilityFunction2(feature_names) / (2 * delta_u))


# 使用指数机制2的概率
def chanceExp2(privacy_budget, c, ci):

    exp_sum = np.sum(exponentialMechanism2(privacy_budget,x) for x in c)

    return exponentialMechanism2(privacy_budget,ci) / exp_sum


# 根据两种效用函数分别选择最佳ci
def selectCi(privacy_budget, c):
    result1 = sorted(zip(map(lambda x: chanceExp1(privacy_budget,c,x), c), c),reverse=True)
    print(result1)
    result2 = sorted(zip(map(lambda x: chanceExp2(privacy_budget,c,x), c), c),reverse=True)
    print(result2)

    return result1[0],result2[0]



# print(utilityFunction1("Outlook"))
selectCi(1,[['Outlook'],['Windy']])
