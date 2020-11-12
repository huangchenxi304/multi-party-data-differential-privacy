import matplotlib.pyplot as plt
import algorithm2_v2
import pandas as pd


# n为指定点个数
def get_xy1(n):
    x = []
    y_ours = []
    y_cs = []
    y_gs = []
    for i in range(n):
        y1, y2, y3 = algorithm2_v2.mae1(i / n)
        y_ours.append(y1)
        y_cs.append(y2)
        y_gs.append(y3)
        x.append(i / n)
    return x, y_ours, y_cs, y_gs


# n为指定点个数
def get_xy2(n):
    x = []
    y_ours = []
    y_cs = []
    y_gs = []
    for i in range(n):
        y1, y2, y3 = algorithm2_v2.mae2(i / n)
        y_ours.append(y1)
        y_cs.append(y2)
        y_gs.append(y3)
        x.append(i / n)
    return x, y_ours, y_cs, y_gs


def painting(x, y_ours, y_cs, y_gs, n, func_name):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title(func_name)
    plt.xlabel('privacy_budget')
    plt.ylabel('MAE')
    plt.plot(x, y_ours)
    plt.plot(x, y_cs)
    plt.plot(x, y_gs)
    plt.legend(['ours', 'cs', 'gs'])
    plt.savefig("n = " + str(n) + "-"+  func_name)
    plt.show()


# 将计算出来的数据存储至本地
def save_data(x, y_ours, y_cs, y_gs, func_name):
    df = pd.DataFrame()
    df['y_ours'] = y_ours
    df['y_cs'] = y_cs
    df['y_gs'] = y_gs
    df['x'] = x
    df.to_excel('%s.xlsx' % func_name)


if __name__ == '__main__':
    point_count = 50
    x, y_ours, y_cs, y_gs = get_xy1(point_count)
    save_data(x, y_ours, y_cs, y_gs, 'u1')
    painting(x, y_ours, y_cs, y_gs, point_count, 'u1')
    x, y_ours, y_cs, y_gs = get_xy2(point_count)
    save_data(x, y_ours, y_cs, y_gs, 'u2')
    painting(x, y_ours, y_cs, y_gs, point_count, 'u2')
