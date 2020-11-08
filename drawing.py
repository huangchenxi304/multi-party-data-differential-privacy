import matplotlib.pyplot as plt
import algorithm2_v2


# n为指定点个数
def get_xy1(n):
    x = []
    y = []
    for i in range(n):
        y.append(algorithm2_v2.mae1(i / n))
        x.append(i / n)
    return x, y


# n为指定点个数
def get_xy2(n):
    x = []
    y = []
    for i in range(n):
        y.append(algorithm2_v2.mae2(i / n))
        x.append(i / n)
    return x, y


def painting(x, y, n, func_name):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.xlabel('privacy_budget')
    plt.ylabel('MAE1')
    plt.plot(x, y)
    plt.savefig("n = " + str(n) + func_name)
    plt.show()


# 将计算出来的数据存储至本地
def save_data(x, y, func_name):
    file_object = open(func_name + '-y.txt', 'w')
    for ip in y:
        file_object.write(str(ip))
        file_object.write('\n')
    file_object.close()
    file_object2 = open(func_name + '-x.txt', 'w')
    for ip in x:
        file_object2.write(str(ip))
        file_object2.write('\n')
    file_object2.close()


if __name__ == '__main__':
    point_count = 50
    x1, y1 = get_xy1(point_count)
    save_data(x1, y1, 'u1')
    painting(x1, y1, point_count, 'u1')
    x2, y2 = get_xy2(point_count)
    save_data(x2, y2, 'u2')
    painting(x2, y2, point_count, 'u2')
