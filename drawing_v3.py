import matplotlib.pyplot as plt
import algorithm2_v3
import pandas as pd


# n为指定点个数
def get_xy1(data_raw, n, adjustment_f, best_f, label, b):
    x = []
    y_ours = []
    y_cs = []
    y_gs = []
    algorithm2_v3.initial_everything1(data_raw, adjustment_f, label, b)
    for i in range(n):
        y1, y2, y3 = algorithm2_v3.mae1(data_raw, i / n, best_f)
        y_ours.append(y1)
        y_cs.append(y2)
        y_gs.append(y3)
        x.append(i / n)
    return x, y_ours, y_cs, y_gs


# n为指定点个数
def get_xy2(data_raw, n, adjustment_f, best_f, label, b):
    x = []
    y_ours = []
    y_cs = []
    y_gs = []
    algorithm2_v3.initial_everything2(data_raw, adjustment_f,label, b)
    for i in range(n):
        y1, y2, y3 = algorithm2_v3.mae2(data_raw, i / n, best_f)
        y_ours.append(y1)
        y_cs.append(y2)
        y_gs.append(y3)
        x.append(i / n)
    return x, y_ours, y_cs, y_gs


def painting(x, y_ours, y_cs, y_gs, n, func_name, data_name, b):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.title('%s-%s-b=%d' % (func_name,data_name,b))
    plt.rcParams['lines.marker'] = '.'
    plt.xlabel('privacy_budget')
    plt.ylabel('MAE')
    plt.plot(x, y_ours)
    plt.plot(x, y_cs)
    plt.plot(x, y_gs)
    plt.legend(['ours', 'cs', 'gs'])
    plt.savefig("n = " + str(n) + "-"+  func_name+"-b="+str(b))
    plt.show()


# 将计算出来的数据存储至本地
def save_data(x, y_ours, y_cs, y_gs, func_name, data_name, b):
    df = pd.DataFrame()
    df['y_ours'] = y_ours
    df['y_cs'] = y_cs
    df['y_gs'] = y_gs
    df['x'] = x
    df.to_excel('%s-%s-b=%d.xlsx' % (func_name,data_name,b))


# 从本地文件读取data
def read_data(path):
    df = pd.read_excel(path)
    y_ours = df['y_ours']
    y_cs = df['y_cs']
    y_gs = df['y_gs']
    x = df['x']
    return y_ours,y_cs,y_gs,x


def run_pic(point_count, data_name, data_raw, adjustment_f, best_f, label, b):

    x, y_ours, y_cs, y_gs = get_xy1(data_raw, point_count, adjustment_f, best_f, label, b)
    save_data(x, y_ours, y_cs, y_gs, 'u1', data_name, b)
    painting(x, y_ours, y_cs, y_gs, point_count, 'u1', data_name,b)
    x, y_ours, y_cs, y_gs = get_xy2(data_raw, point_count, adjustment_f, best_f, label, b)
    save_data(x, y_ours, y_cs, y_gs, 'u2', data_name, b)
    painting(x, y_ours, y_cs, y_gs, point_count, 'u2', data_name, b)


y_ours,y_cs,y_gs,x = read_data('u2-titanic-b=3.xlsx')
painting(x, y_ours, y_cs, y_gs, 10, 'u2', 'titanic', 3)