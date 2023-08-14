from scipy.integrate import odeint
from scipy import integrate
import numpy as np
from sko.GA import GA

def func(z, t, f, w, k1, k2, k3, M, M_prime, m, rho, g, R, alpha):
    x, dxdt, y, dydt = z
    d2xdt2 = (- k1 * (x - y) - k2 * (dxdt - dydt) * (abs(dxdt - dydt) ** alpha)) / m
    d2ydt2 = (f * np.cos(w * t) + k1 * (x - y) + k2 * (dxdt - dydt) * (abs(dxdt - dydt) ** alpha) - k3 * dydt - rho * g * np.pi * (R ** 2) * y) / (M + M_prime)
    return [dxdt, d2xdt2, dydt, d2ydt2]


def j2P(j, a):
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
    alpha = a
    P = []
    # 求解方程组
    sol = odeint(func, y0, t, args=(f, w, k1, k2, k3, M, M_prime, m, rho, g, R, alpha))
    for i in t[len(t) // 5 * 4:]:
        p = (abs(sol[int(i*1000)][1] - sol[int(i*1000)][3]) ** (2 + a)) * j
        P.append(p)
    v = integrate.trapz(P, t[len(t) // 5 * 4:])

    return v / 30


def fun(solution):
    return -j2P(solution[0], solution[1])



ga = GA(func=fun, n_dim=2, prob_mut=0.01, size_pop=100, max_iter=50, lb=[0, 0], ub=[100000, 1])

best_ja, best_P = ga.run()
print('best_j:', best_ja[0], '\n','best_a:', best_ja[1], 'best_P:', -best_P)

# runtime: 12 min
# best_j : 99999.65
# best_a : 0.4126
# best_P : 229.71