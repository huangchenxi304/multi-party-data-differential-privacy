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
    print('--------------------------')
    for i in range(1,n+1):
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
    for i in range(1,n+1):
        y1, y2, y3 = algorithm2_v3.mae2(data_raw, i / n, best_f)
        y_ours.append(y1)
        y_cs.append(y2)
        y_gs.append(y3)
        x.append(i / n)
    return x, y_ours, y_cs, y_gs


def painting(x, y_ours, y_cs, y_gs, n, func_name, data_name, b):
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.title('%s-%s-b=%d' % (func_name,data_name,b))
    plt.rcParams['lines.marker'] = '.'
    plt.grid()  # 生成网格
    plt.xlabel('Epslion')
    plt.ylabel('MAE')
    plt.xticks(x)
    plt.plot(x, y_ours)
    plt.plot(x, y_cs)
    plt.plot(x, y_gs)
    plt.legend(['Private CSₚ', 'Private CS', 'Private GS'])
    plt.savefig('%s-%s-b=%d' % (func_name,data_name,b))
    plt.show()


def painting_on_1picture(dicts):
    # 全局设置
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['lines.marker'] = '.'
    plt.rcParams['axes.grid'] = True

    p = plt.figure(figsize=(8, 6), dpi=80)  ## 确定画布大小
    k=0
    for i in dicts:
        k += 1
        p.add_subplot(2, 2, k) ## 创建一个2行2列的子图，并开始绘制第k幅
        plt.title('')
        plt.xlabel('Epslion')
        plt.ylabel('MAE')
        x = i['x']
        plt.xticks(x)
        plt.plot(x, i['y_ours'])
        plt.plot(x, i['y_cs'])
        plt.plot(x, i['y_gs'])
        plt.legend(['Private CSₚ', 'Private CS', 'Private GS'])
    plt.savefig('----')
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
    dict = {'y_ours':y_ours,'y_cs':y_cs,'y_gs':y_gs,'x':x}
    return dict


def run_pic(point_count, data_name, data_raw, adjustment_f, best_f, label, b):

    x, y_ours, y_cs, y_gs = get_xy1(data_raw, point_count, adjustment_f, best_f, label, b)
    save_data(x, y_ours, y_cs, y_gs, 'u1', data_name, b)
    painting(x, y_ours, y_cs, y_gs, point_count, 'u1', data_name,b)
    x, y_ours, y_cs, y_gs = get_xy2(data_raw, point_count, adjustment_f, best_f, label, b)
    save_data(x, y_ours, y_cs, y_gs, 'u2', data_name, b)
    painting(x, y_ours, y_cs, y_gs, point_count, 'u2', data_name, b)


dict1 = read_data('u1-adult-b=2.xlsx')
dict2 = read_data('u2-adult-b=2.xlsx')
dict3 = read_data('u1-titanic-b=2.xlsx')
dict4 = read_data('u2-titanic-b=4.xlsx')

dicts = [dict1, dict2, dict3, dict4]
painting_on_1picture(dicts)