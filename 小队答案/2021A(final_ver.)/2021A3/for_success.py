import numpy as np
import pandas as pd
import random
import winsound
alpha = np.deg2rad(36.795)
gamma = np.deg2rad(90 - 78.169)
# 计算旋转矩阵
Rz = np.array([[np.cos(alpha),  np.sin(alpha), 0],
               [-np.sin(alpha), np.cos(alpha), 0],
               [0,              0,             1]])

Ry = np.array([[np.cos(gamma), 0, -np.sin(gamma)],
               [0,             1,              0],
               [np.sin(gamma), 0,  np.cos(gamma)]])

R = Ry @ Rz
position_dict = {}
position_dict_strict = {}

# 基准态球面
data1 = pd.read_csv('success.csv', usecols=range(0, 4), encoding="gbk")
data3 = pd.read_csv('data3.csv', usecols=range(0, 3), encoding="gbk")

for row in data1.iloc:
    x0 = row[1]
    y0 = row[2]
    z0 = row[3]
    # 求出每点在新坐标系下的坐标，并保存至a_dict
    A_0 = [[x0], [y0], [z0]]
    # 去掉内层中括号
    A_1 = [A_0[0][0], A_0[1][0], A_0[2][0]]
    # 取处在工作态及附近的点
    if x0**2+y0**2 <= 180**2:
        position_dict[row[0]] = A_1
    if x0**2+y0**2 <= 150**2:
        position_dict_strict[row[0]] = A_1

triangle_dict = {}
i = 0
for row in data3.iloc:
    A = row[0]
    B = row[1]
    C = row[2]

    if (A in position_dict_strict) or (B in position_dict_strict) or (C in position_dict_strict):
        A_position = position_dict[A]
        B_position = position_dict[B]
        C_position = position_dict[C]
        # 求三角形重心
        G = [(A_position[0] + B_position[0] + C_position[0])/3, (A_position[1] + B_position[1] + C_position[1]) /
             3, (A_position[2] + B_position[2] + C_position[2])/3]
        # 求沿z轴正方向的法向量

        VEC_AB = np.array([B_position[0]-A_position[0],
                           B_position[1]-A_position[1],
                           B_position[2]-A_position[2]])
        VEC_AC = np.array([C_position[0]-A_position[0],
                           C_position[1]-A_position[1],
                           C_position[2]-A_position[2]])
        n = np.cross(VEC_AB, VEC_AC)
        if (n[2] < 0):
            n = [-n[0], -n[1], -n[2]]
        else:
            n = [n[0], n[1], n[2]]
        # 将A\B\C和法向量存入三角形字典
        triangle_dict[i] = [A_position,B_position,C_position, n]
        i = i + 1

# 根据蒙特卡罗法，随机生成位于工作态内的坐标(x, y)
# 记录距离随机点最近的三角形
triangle_close = []
righttimes = 0
wrongtimes = 0
# 设置蒙特卡洛实验次数
mc_times = 1000000
for j in range(0, mc_times):
    print(j)
    dis = np.inf
    x = random.randrange(-150, 150)
    y = random.randrange(-150, 150)

    if ((x**2 + y**2) > 150**2):
        continue
    for num, tri in triangle_dict.items():
        u = x - tri[0][0]
        v = y - tri[0][1]
        m = tri[1][0] - tri[0][0]
        n = tri[1][1] - tri[0][1]
        r = tri[2][0] - tri[0][0]
        s = tri[2][1] - tri[0][1]
        a = (s * u - r * v) / (s * m - r * n)
        b = (n * u - m * v) / (n * r - m * s)
        if(a > 0) and (b > 0) and (a + b < 1):
            triangle_close = tri
            break
    A_close = triangle_close[0]
    n_close = triangle_close[3]
    z = (A_close[0] * n_close[0] + A_close[1] * n_close[1] +
         A_close[2] * n_close[2] - n_close[0]*x - n_close[1]*y)/n_close[2]
    # 计算光线投影在馈源舱平面的坐标

    x1 = n_close[0]
    y1 = n_close[1]
    z1 = n_close[2]
    z_prime = abs(z)-0.534 * 300.4
    x_reflect = x + 2 * x1 * z1*z_prime/(z1*z1-(x1*x1+y1*y1))
    y_reflect = y + 2 * y1 * z1*z_prime/(z1*z1-(x1*x1+y1*y1))


    if (x_reflect**2+y_reflect**2) <= 0.5**2:
        righttimes = righttimes + 1
    else:
        wrongtimes = wrongtimes + 1
print(righttimes / (righttimes + wrongtimes))
