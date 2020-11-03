from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


# 数据数字化
def dataDigitize(path):
    # 获得原始数据
    adult_raw = pd.read_excel(path)

    # 清理数据，删除缺失值
    adult_cleaned = adult_raw.dropna()

    # 将bool变量数字化
    if 'Windy' in adult_cleaned.columns:

        for i in range(adult_cleaned['Windy'].shape[0]):
            adult_cleaned.loc[i, 'Windy'] = adult_cleaned.loc[i, 'Windy'] + 0

    # 其他属性数字化
    adult_digitization = pd.DataFrame()

    target_columns = adult_cleaned.columns
    for column in adult_cleaned.columns:
        if column in target_columns:

            unique_value = list(enumerate(np.unique(adult_cleaned[column])))
            dict_data = {key: value for value, key in unique_value}
            adult_digitization[column] = adult_cleaned[column].map(dict_data)
        else:
            adult_digitization[column] = adult_cleaned[column]



    return adult_digitization

# 参数：data——处理后的数据；adult_clf——选用的训练模型；name——指定的模型名称
def main(data,labels,adult_clf, name=""):
    adult_digitization = data


    # 构造输入和输出
    X = adult_digitization
    Y = labels

    # 交叉验证
    preaccsvm = []
    num = 1
    kf = KFold(n_splits=10)

    for train, test in kf.split(X):
        X_train, X_test = X.loc[train], X.loc[test]
        Y_train, Y_test = Y.loc[train], Y.loc[test]

        adult_clf.fit(X_train, Y_train.values.ravel())

        # test_score = clf.score(X_test, Y_test)
        # print("test_score:" + str(test_score))
        test_predictions = adult_clf.predict(X_test)
        accuracy = accuracy_score(Y_test.values.ravel(), test_predictions)
        preaccsvm.append(accuracy)
        # print(name + str(num) + "测试集准确率:  %s " % accuracy)
        num = num + 1

    # print(name + "十折交叉平均准确率:  %s " % np.mean(np.array(preaccsvm)))

    # 返回十折交叉平均准确率
    return np.mean(np.array(preaccsvm))

def runSvm(data,labels):
    svmclf = svm.SVC(kernel='rbf', C=1)
    return main(data,labels,svmclf,'svm')


MNBclf = MultinomialNB()
# main(MNBclf,'MNB')

GNBclf = GaussianNB()
# main(GNBclf,'GNB')

BNBclf = BernoulliNB()
# main(BNBclf,'BNB')

# dataPartition(3)


