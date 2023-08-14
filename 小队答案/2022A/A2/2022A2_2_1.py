from scipy.integrate import odeint
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

# 基于变步长的最优搜索（双变量，单目标）
def func(z, t, f, w, k1, k2, k3, M, M_prime, m, rho, g, R, alpha):
    x, dxdt, y, dydt = z
    d2xdt2 = (- k1 * (x - y) - k2 * (dxdt - dydt) * (abs(dxdt - dydt) ** alpha)) / m
    d2ydt2 = (f * np.cos(w * t) + k1 * (x - y) + k2 * (dxdt - dydt) * (abs(dxdt - dydt) ** alpha) - k3 * dydt - rho * g * np.pi * (R ** 2) * y) / (M + M_prime)
    return [dxdt, d2xdt2, dydt, d2ydt2]

ja_dict = {}


# 变步长搜索
#              0, 100000, 1000
#              80000, 100000, 100
#              99000, 100000, 10
for j in range(0, 100000, 1000):
    print(j)
    # 定义初始条件和时间范围    
    y0 = [0.0, 0.0, 0.0, 0.0]
    t = np.linspace(0, 150, 150001)

    # 定义参数
    f = 4890
    w = 2.2143
    k1 = 80000
    k2 = j
    k3 = 167.8395
    M = 4866
    m = 2433
    M_prime = 1165.992
    rho = 1025
    g = 9.8
    R = 1
#                      0, 1, 0.1
#                      0.35, 0.45, 0.01 
#                      0.415, 0.425, 0.001
    for a in np.arange(0, 1, 0.1):

        alpha = a 
        P = []
        # 求解方程组
        sol = odeint(func, y0, t, args=(f, w, k1, k2, k3, M, M_prime, m, rho, g, R, alpha))
        for i in t[len(t) //3 * 2:]:
            p = (abs(sol[int(i*1000)][1] - sol[int(i*1000)][3]) ** (2 + alpha)) * j
            P.append(p)
        v = integrate.trapz(P, t[len(t) //3 * 2:])
        ja_dict[(j, a)] = v / 50

print(max(ja_dict, key=ja_dict.get))


# 三维图像绘制

fig = plt.figure()
ax3 = plt.axes(projection='3d')

jj = []
aa = []

for m in ja_dict:
    jj.append(m[0])
    aa.append(m[1])

J, A = np.meshgrid(jj, aa)
list_out = []
for aa_i in aa:
    list_in = []
    for jj_i in jj:
        list_in.append(ja_dict[(jj_i, aa_i)])
    list_out.append(list_in)
P = np.array(list_out)

ax3.plot_surface(J, A, P, cmap='rainbow')
plt.show()