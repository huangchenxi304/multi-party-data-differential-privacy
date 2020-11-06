import pandas as pd
import numpy as np



def noise_count_error(data: pd.DataFrame, delta_f, epsl) -> float:
    """ 输入dataframe与隐私参数计算MAE
    """
    # 拉普拉斯参数
    scale = delta_f / epsl
    # 查询次数
    num = 100

    error_sum = 0
    # 遍历每列
    for col in data.columns:

        true_count = list(data[col].value_counts().values)

        # 重复查询求该列平均error
        for x in range(num):
            # 计算随机查询目标的error
            # 当前公式下，error = laplace random
            query_target = np.random.randint(0, len(true_count))
            noise_count = true_count[query_target] + np.random.laplace(0, scale, 1)[0]
            error = abs(noise_count - true_count[query_target])
            error_sum += error

    return error_sum / (data.columns.size * num)


