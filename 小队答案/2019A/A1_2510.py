import numpy as np
import csv
from scipy import interpolate
import pylab as plt


# 得出关于rho的P函数
P = []
E_inverse = []

with open("data1.csv", "r") as csvfile:
    reader = csv.reader(csvfile)  # 创建一个reader对象
    for row in reader:
        P.append(float(row[0]))
        E_inverse.append(1/float(row[1]))

print()

rho_list = []

for i in range(0, 200):
    rho_list.append(0.850 * np.exp(-np.trapz(E_inverse[i: 201], P[i: 201])))

rho_list.append(0.850)

for i in range(202, 402):
    rho_list.append(0.850 * np.exp(np.trapz(E_inverse[200: i], P[200: i])))


rho_new = np.arange(0.8043, 0.8825, 0.0001)

coefficients = np.polyfit(rho_list, P, 10)

P_new = np.polyval(coefficients, rho_new)


# tck = interpolate.splrep(rho_list, P)
# P_bspline = interpolate.splev(rho_new, tck)


# plt.rc('text', usetex=True)
# plt.rc('font', size=15)
# plt.xlabel(r'$\rho$')
# plt.ylabel('$P$', rotation=0)
# plt.plot(rho, P, 'o')  # 显示原始点数据
# plt.plot(rho_new, P_bspline, label="Bspline_IP")  # 显示样条插值
# plt.legend()
# plt.show()

# Prho_dict = {}
# for i in range(0, len(rho_new)):
#     Prho_dict[round(rho_new[i],4)] = P_bspline[i]



# 参数表
rho0 = 0.8711223
L = 500
D = 10
d = 1.4
V = np.pi * (D/2)**2 * L
P1 = 160
C = 0.85
A = np.pi * (d/2)**2
t0 = 10

# 供油入口流量
def Q_in(rho, T, t, P):
    t_temp = t % (T + t0)
    # 开启时
    if t_temp < T:
        return C * A * np.sqrt(2 * (P1 - P) / rho)
    else:
        return 0

# 喷油口流量
def Q_out(t):
    t_temp = t % 100
    # 喷射态
    if t_temp < 0.2:
        return 100 * t_temp
    elif t_temp < 2.2:
        return 20
    elif t_temp < 2.4:
        return 20-100 * (t_temp - 2.2)
    else:
        return 0

# 偏差值总和列表
Ts_dict = {}

# 搜索范围0-20   步长为0.1
for T in np.arange(0.74, 0.76, 0.002):
    print(T)
    # 重置参数
    dt = 0.01
    t = 0
    # rho = 0.86793   # 150Pa
    rho = 0.850       # 100Pa
    flag = 0
    # 存放每个时间点对应的P值
    tP_dict = {}
    while t <= 10010:
        if(rho < rho_new[0]) or (rho > rho_new[-1]):
            flag = 1
            break
        P_temp = np.polyval(coefficients, rho)
        tP_dict[t] = P_temp
        rho_prime = (Q_in(rho, T, t, P_temp) * rho0 - Q_out(t) * rho)/V
        rho += rho_prime * dt
        t += dt

    if flag == 1:
        continue
    s = 0

    # 取一个点运算的结果不够准确， 应该取周围的一个邻域
    for i in range(0, 2000):
        t -= dt
        s += tP_dict[t]
    Ts_dict[T] = abs(s/2000 - 150)

Key = []
Value = []
for k,v in Ts_dict.items():
    Key.append(k)
    Value.append(v)


plt.rc('text', usetex= True)
plt.rc('font', size= 15)
plt.xlabel('$ T $')
plt.ylabel('$ s $', rotation = 0)
plt.plot(Key, Value, 'o')  # 显示原始点数据
plt.show()


