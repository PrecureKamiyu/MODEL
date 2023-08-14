from scipy.integrate import odeint
from scipy import integrate
import numpy as np
from sko.GA import GA

# 基于遗传算法的最优条件搜索

def func(z, t, f, w, k1, k2, k3, M, M_prime, m, rho, g, R):
    x, dxdt, y, dydt = z
    d2xdt2 = (- k1 * (x - y) - k2 * (dxdt - dydt)) / m
    d2ydt2 = (f * np.cos(w * t) + k1 * (x - y) + k2 * (dxdt - dydt) - k3 * dydt - rho * g * np.pi * (R ** 2) * y) / (M + M_prime)
    return [dxdt, d2xdt2, dydt, d2ydt2]

# 定义映射函数
def j2P(j):
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

    P = []
    sol = odeint(func, y0, t, args=(f, w, k1, k2, k3, M, M_prime, m, rho, g, R))
    for i in t[len(t) // 5 * 4:]:
        p = ((sol[int(i*1000)][1] - sol[int(i*1000)][3]) ** 2) * j
        P.append(p)
    v = integrate.trapz(P, t[len(t) // 5 * 4:])

    return v / 30

# >>>>>>>>
def fun(solution):
    return -j2P(solution[0])

# 遗传算法的最优指的是取到最小值
#       func>>>>, n_dim变量数  prob_mut变异概率  size_pop种群组数量   max_iter迭代次数   ub lb上下限
ga = GA(func=fun, n_dim=1,    prob_mut=0.01,    size_pop=100,       max_iter=50, lb=[0], ub=[100000])

# ga.run()方法得到最优解
best_j, best_P = ga.run()
print('best_j:', best_j, '\n', 'best_P:', best_P)

# runtime: 6min
# best_j : 37299
# best_P : -228.81