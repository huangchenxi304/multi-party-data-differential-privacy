import itertools
import algo_2_count
import numpy as np
import funcs
import warnings

warnings.filterwarnings("ignore")


# 根据调整特征系数对调整特征集的特征进行组合，生成备选方案
def partition_c(adjustment_features, label):
    # 两两组合，返回备选方案c
    c_raw = [list(i) for i in itertools.combinations(adjustment_features, 2)]
    for x in c_raw:
        x.append(label)
    return c_raw


# 计算信息增益
def gain_info(data_raw, feature_name, label):
    # 合并后数据集Dp的初始信息熵
    info_entropy_dp = cal_info_entropy(data_raw, label)

    # 根据特征名称获取特征取值的count
    count_a_feature = data_raw[feature_name].value_counts()

    # 根据特征取值划分子集
    subset = [data_raw.loc[data_raw[feature_name] == value] for value in count_a_feature._index]

    # 计算各子集的信息熵
    subset_entropy = [cal_info_entropy(x, label) for x in subset]

    feature_chance = np.array([x / np.sum(count_a_feature) for x in count_a_feature])

    # 计算信息增益
    return info_entropy_dp - np.matmul(feature_chance, subset_entropy)


# 计算信息熵
def cal_info_entropy(data, data_label):
    # 计算标签取值count情况
    label_data = data[data_label].value_counts()
    # 计算标签取值概率
    chance = [x / np.sum(label_data) for x in label_data]
    # 计算信息熵
    info_entropy = -(np.sum(np.fromiter((x * np.log2(x) for x in chance), float)))
    return info_entropy


# 计算单个特征的信息增益的字典
def gain_info_initialize(adjustment_features, label, data_raw):
    single_feature_gain_info = dict(
        zip(adjustment_features, map(lambda x: gain_info(data_raw, x, label), adjustment_features)))
    single_feature_gain_info[label] = 0
    return single_feature_gain_info


def u1_initialize(c, gain_info_dict):
    # 将备选方案中所有子列表转元组，因为list是unhashable
    tuple_c = tuple(tuple(x) for x in c)
    # 备选方案的信息增益为 b 个特征增益的和
    sum_gain_info = []
    for x in c:
        sum_gain_info.append(np.sum(gain_info_dict[feature] for feature in x))
    return dict(zip(tuple_c, sum_gain_info))


# 效用函数1
def utility_function1(feature_names, u1_dict):
    return u1_dict[tuple(feature_names)]


# n方数据集平均相关度
def get_mcd(data_raw):
    return np.mean([funcs.cs(x)['CS_mean'] for x in funcs.split(data_raw, 3)])


# 计算u2字典
def u2_initialize(data_raw, c, mcd):
    # 将备选方案中所有子列表转元组，因为list是unhashable
    tuple_c = tuple(tuple(x) for x in c)
    return dict(zip(tuple_c, map(lambda x: funcs.cs(data_raw[x], mcd)['CS_i'], c)))


# 效用函数2
def utility_function2(feature_names, mcd, u2_dict):
    return mcd / u2_dict[tuple(feature_names)]


# 使用效用函数1的指数机制1
def exponential_mechanism1(privacy_budget, feature_names, u1_dict, delta_u=1, ):
    return np.exp(privacy_budget * utility_function1(feature_names, u1_dict) / (2 * delta_u))


# 使用指数机制1的概率
def chance_exp1(privacy_budget, c, ci, u1_dict):
    exp_sum = np.sum(exponential_mechanism1(privacy_budget, x, u1_dict) for x in c)

    return exponential_mechanism1(privacy_budget, ci, u1_dict) / exp_sum


# 使用效用函数2的指数机制
def exponential_mechanism2(privacy_budget, feature_names, mcd, u2_dict, delta_u=1):
    return np.exp(privacy_budget * utility_function2(feature_names, mcd, u2_dict) / (2 * delta_u))


# 使用指数机制2的概率
def chance_exp2(privacy_budget, c, ci, mcd, u2_dict):
    exp_sum = np.sum(exponential_mechanism2(privacy_budget, x, mcd, u2_dict) for x in c)

    return exponential_mechanism2(privacy_budget, ci, mcd, u2_dict) / exp_sum


def select_ci1(privacy_budget, c, best_features, u1_dict):
    # 根据效用函数1选出的ci
    selected_features1 = sorted(zip(map(lambda x: chance_exp1(privacy_budget, c, x, u1_dict), c), c), reverse=True)
    # 将选出的最佳ci加入best features
    return best_features + list(selected_features1[0])[1]


def select_ci2(privacy_budget, c, best_features, mcd, u2_dict):
    # 根据效用函数2选出的ci
    selected_features2 = sorted(zip(map(lambda x: chance_exp2(privacy_budget, c, x, mcd, u2_dict), c), c), reverse=True)

    # 将选出的最佳ci加入best features
    return best_features + list(selected_features2[0])[1]


# 打印lyc要的cs
def print_cs(name, data, threshold=0.5):
    print(name + ':(阈值为 %s )' % threshold)
    print(data.columns)
    print(funcs.cs(data, threshold))


def initial_everything1(data_raw, adjustment_f, label):
    global backup_solutions,gain_info_dict,u1_dict,mcd
    # 生成备选方案c
    backup_solutions = partition_c(adjustment_f, label)
    gain_info_dict = gain_info_initialize(adjustment_f, label, data_raw)
    u1_dict = u1_initialize(backup_solutions, gain_info_dict)
    mcd = get_mcd(data_raw)


def initial_everything2(data_raw, adjustment_f, label):
    global backup_solutions,u2_dict,mcd
    # 生成备选方案c
    backup_solutions = partition_c(adjustment_f, label)
    mcd = get_mcd(data_raw)

    # 生成u2字典
    u2_dict = u2_initialize(data_raw, backup_solutions, mcd)

def mae1(data_raw, privacy_budget, best_f):



    best_and_selected1 = select_ci1(1, backup_solutions, best_f, u1_dict)
    # print_cs('best+u1选出来的特征', data_raw[best_and_selected1])
    # print_cs('best+u1选出来的特征', data_raw[best_and_selected1], float(mcd))

    u1 = algo_2_count.noise_count_error(data_raw[best_and_selected1],
                                        funcs.cs(data_raw[best_and_selected1], mcd)['CS_i'],
                                        privacy_budget)

    # cs,阈值为默认0.5
    mae_cs = algo_2_count.noise_count_error(data_raw[best_and_selected1],
                                            funcs.cs(data_raw[best_and_selected1])['CS_i'],
                                            privacy_budget)
    # GS，阈值为默认0.5
    mae_gs = algo_2_count.noise_count_error(data_raw[best_and_selected1],
                                            funcs.cs(data_raw[best_and_selected1])['GS'],
                                            privacy_budget)
    print('u1:' + str(u1))
    print('u1cs:' + str(mae_cs))
    print('u1gs:' + str(mae_gs))

    return u1, mae_cs, mae_gs


def mae2(data_raw, privacy_budget, best_f):


    best_and_selected2 = select_ci2(1, backup_solutions, best_f, mcd, u2_dict)

    # print_cs('best+u2选出来的特征', data_raw[best_and_selected2])
    # print_cs('best+u2选出来的特征', data_raw[best_and_selected2], float(mcd))

    # 我们的算法，使用mcd为阈值
    u2 = algo_2_count.noise_count_error(data_raw[best_and_selected2],
                                        funcs.cs(data_raw[best_and_selected2], mcd)['CS_i'],
                                        privacy_budget)

    # cs,阈值为默认0.5
    mae_cs = algo_2_count.noise_count_error(data_raw[best_and_selected2],
                                            funcs.cs(data_raw[best_and_selected2])['CS_i'],
                                            privacy_budget)
    # GS，阈值为默认0.5
    mae_gs = algo_2_count.noise_count_error(data_raw[best_and_selected2],
                                            funcs.cs(data_raw[best_and_selected2])['GS'],
                                            privacy_budget)
    print('u2:' + str(u2))
    print('u2cs:' + str(mae_cs))
    print('u2gs:' + str(mae_gs))

    return u2, mae_cs, mae_gs
