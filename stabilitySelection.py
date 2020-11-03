import d_SVM
import numpy as np
import pandas as pd
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


def getData(path):
    global data, labels
    data = d_SVM.dataDigitize(path)
    labels = data['Play?']
    data.drop("Play?", axis=1, inplace=True)


def crossValidationPreparation(f):
    new_data = data.__deepcopy__()

    for column in data.columns:
        if column not in f:
            new_data.drop(column, axis=1, inplace=True)

    return new_data


# 此方法用0.19.2版本的sklearn和0.23.2版本跑出来的结果不一样
def mySelection():
    # 初始化特征集
    features = []

    # 自定义阈值
    threshold = 6
    best_score = 0
    best_features = []

    print("选择前精度：")
    print(d_SVM.runSvm(data, labels))

    while len(features) < threshold:

        for i in range(len(data.columns)):
            if data.columns[i] not in features:
                features.append(data.columns[i])
                new_data = crossValidationPreparation(features)
                accuracy = d_SVM.runSvm(new_data, labels)
                print(features)
                print(accuracy)

                if accuracy > best_score:
                    best_score = accuracy
                    best_features = features
                    # print(best_features)
                    # print(best_score)
    # print("最佳特征集：")
    # print(best_features)
    # print("选择后精度：")
    # print(d_SVM.runSvm(data[['Outlook', 'health', 'Humidity']],labels))


def rlassoSS():
    global data, labels
    names = data.columns
    rlasso = RandomizedLasso(alpha=0.025)
    rlasso.fit(data, labels)

    print("Features sorted by their score:")
    result = sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                        names), reverse=True)
    for i in result:
        print(i)


def recursiveFeatureElimination():
    global data, labels
    names = data.columns
    # use linear regression as the model
    lr = LinearRegression()
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(data, labels)

    print("Features sorted by their rank:")
    result = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
    for i in result:
        print(i)


# 计算记录相关度矩阵
def calRecordCorr():
    # 创建空矩阵
    n = data.shape[0]
    matrix = pd.DataFrame([[0] * n for i in range(n)])
    # 按行遍历
    for index, row in data.iterrows():

        # 逐一计算当前行与其之后行的记录相关度
        for i in range(index + 1, data.shape[0]):
            mach = 0
            # 按特征从左到右遍历
            for col_name in data.columns:
                # 若两条记录的col_name特征取值相同，则mach+1
                if row[col_name] == data.loc[index + 1, col_name]:
                    mach = mach + 1
            # 计算记录相关度
            w_i_j = mach / data.shape[1]
            # 存储到相关度矩阵中
            matrix.loc[index, i] = w_i_j
    print(matrix)


getData("D:\学习\差分隐私\d1.xlsx")
# 用pearson相关系数来计算记录相关性
# c = data.transpose().corr()
# print(c)
calRecordCorr()

# rlassoSS()
# getData("D:\学习\差分隐私\d2.xlsx")
# rlassoSS()
# mySelection()
# recursiveFeatureElimination()
