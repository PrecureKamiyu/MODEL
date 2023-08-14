from scipy.integrate import odeint
from scipy import integrate
import numpy as np
import pylab as plt

# 基于变步长的最优搜索（单变量，单目标）
# 以下介绍odeint方法
# odeint方程函数
def func(z, t, f, w, k1, k2, k3, M, M_prime, m, rho, g, R):
    #  （待求项） = 参数
    x, dxdt, y, dydt = z
    #  列方程
    d2xdt2 = (- k1 * (x - y) - k2 * (dxdt - dydt)) / m
    d2ydt2 = (f * np.cos(w * t) + k1 * (x - y) + k2 * (dxdt - dydt) - k3 * dydt - rho * g * np.pi * (R ** 2) * y) / (M + M_prime)
    #  返回高阶导项
    return [dxdt, d2xdt2, dydt, d2ydt2]

j_dict = {}

# trapz梯形方法求积分，先把自变量区间细分
t = np.linspace(0, 150, 150001)

# 搜索区间及变步长
#              0, 100000, 100
#              30000, 40000, 10
#              36000, 38000, 1
for j in range(0, 100000, 100):
    print(j)
    # 定义参数（每一问的参数可能是不一样的）
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

    # odeint定义（待求项）初始条件和时间范围
    y0 = [0.0, 0.0, 0.0, 0.0]

    P = []
    # 求解方程组
    # sol为待求项组的集合  sol = odeint(func, 待求项组, 自变量, ...)
    sol = odeint(func, y0, t, args=(f, w, k1, k2, k3, M, M_prime, m, rho, g, R))

    # 为减小初值条件对于稳定状态的影响，取[0:150]中的[120:150]进行分析
    for i in t[len(t) //5 * 4:]:
        p = ((sol[int(i*1000)][1] - sol[int(i*1000)][3]) ** 2) * j
        P.append(p)

    # x = [] , y = y(x) = [] trapz(y,x)法求积分
    v = integrate.trapz(P, t[len(t) //5 * 4:])

    j_dict[j] = v / 30  #  除以时间长度

print(max(j_dict, key=j_dict.get))

x = []
y = []

for key in j_dict:
    x.append(key)
    y.append(j_dict.get(key))

plt.plot(x, y)
plt.show()