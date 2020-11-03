import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from randomized_l1 import RandomizedLasso


def cs(data, threshold=0.5):
    """
    求cs
    :param data: Dataframe
    :param threshold: 阈值，默认为0.5
    :return: 一个Series，包括
    """
    # Dataframe转Array
    data_value = data.values
    rows = data_value.shape[0]
    # 相关度矩阵
    mat = np.zeros((rows, rows), dtype='float32')
    for i in range(rows):
        data_eql = np.equal(data_value[i], data_value)
        for j in range(i + 1, rows):
            mat[i][j] = np.sum(data_eql[j]) / data_value.shape[1]
    mat += mat.T
    # 每行符合阈值条件个数的数组
    k = np.sum(mat > threshold, axis=0)
    # 每行符合阈值条件的数之和的数组
    w = np.sum(np.ma.MaskedArray(mat, mask=(mat <= threshold)), axis=0)
    # 生成输出
    output = pd.Series([np.max(w), np.max(w) / k[np.argmax(w)], k[np.argmax(w)]], index=['CS_i', 'CS_mean', 'GS'], dtype='float32')
    return output

def randomized_lasso(x: np.ndarray, y: list, names: list, threshold=0, best_features=0):
    """
    随机lasso算法
    :param x: 数据集
    :param y: 标签
    :param names: 不包含标签的列名称
    :param threshold: 阈值
    :param best_features: 重要特征的数量
    :return: 重要特征集和调整特征集的列名称
    """
    # RandomizedLasso
    rlasso = RandomizedLasso(verbose=True)
    rlasso.fit(x, y)
    best_feature_set_names = []  # 重要特征集（列标签）
    adjusted_feature_set_names = []  # 调整特征集（列标签）
    result = sorted(zip(map(lambda ii: round(ii, len(names)), rlasso.scores_), names), reverse=True)
    if threshold > 0:
        for j in result:
            if j[0] >= threshold:
                best_feature_set_names.append(j[1])
                print('best feature:', j[0], j[1], sep=' ')
            else:
                adjusted_feature_set_names.append(j[1])
                print('adjusted feature:', j[0], j[1], sep=' ')
    elif best_features > 0:
        for i in result[:best_features]:
            print('best feature:', i[0], i[1], sep=' ')
            best_feature_set_names.append(i[1])
        for i in result[best_features:]:
            adjusted_feature_set_names.append(i[1])
            print('adjusted feature:', i[0], i[1], sep=' ')
    else:
        raise ValueError('需指定阈值或重要特征数量')
    return best_feature_set_names, adjusted_feature_set_names


def data_clean(data: pd.DataFrame):
    """
    清洗数据
    :param data: 一个DataFrame
    :return: 清洗后的DataFrame
    """
    # 数据标签分离
    ad_y = data.iloc[:, -1]
    ad_x = data.iloc[:, 0:-1]
    # 处理布尔类型
    for i in range(ad_x.shape[1]):
        if ad_x.iloc[:, i].dtype == 'bool':
            ad_x.iloc[:, i] = ad_x.iloc[:, i].astype('int')
    # 处理哑变量
    ad_x = pd.get_dummies(ad_x)
    ad_y = LabelEncoder().fit_transform(ad_y)

    names = list(ad_x.columns.values)
    return ad_x, ad_y, names


def cast_features(best_feature_names: list, adjusted_feature_names: list, sep='_'):
    """
    恢复哑变量处理后的特征名称并去重
    :param best_feature_names: 重要特征名称
    :param adjusted_feature_names: 调整特征名称
    :param sep: 分隔符
    :return: 处理后的特征名称
    """
    best_features = []
    adjusted_features = []
    for i in best_feature_names:
        if len(i) > 1:
            best_features.append(i.split(sep)[0])
        else:
            best_features.append(i[0])
    for i in adjusted_feature_names:
        if len(i) > 1:
            adjusted_features.append(i.split(sep)[0])
        else:
            adjusted_features.append(i[0])
    best_features = list(set(best_features))
    adjusted_features = list(set(adjusted_features).difference(set(best_features)))
    return best_features, adjusted_features
