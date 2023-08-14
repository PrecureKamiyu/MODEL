import numpy as np


# 定义参数
f = 3640
w = 1.7152
k1 = 80000
k2 = 10000
k3 = 683.4558
M = 4866
m = 2433
M_prime = 1028.876
p = 1025
g = 9.8
R = 1


def func(t, z):
    x, dxdt, y, dydt = z
    d2xdt2 = (- k1 * (x - y) - k2 * (dxdt - dydt)) / m
    d2ydt2 = (f * np.cos(w * t) + k1 * (x - y) + k2 * (dxdt - dydt) - k3 * dydt - p * g * np.pi * (R ** 2) * y) / (M + M_prime)
    return np.array([dxdt, d2xdt2, dydt, d2ydt2])



def rk4_step(fun, x, y, h):
    # Compute one step of the Runge-Kutta 4 method
    k1 = h * fun(x, y)
    k2 = h * fun(x + h / 2, y + k1 / 2)
    k3 = h * fun(x + h / 2, y + k2 / 2)
    k4 = h * fun(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def rk4_solve(fun, x0, xf, y0, h):
    # Solve the system of differential equations using the Runge-Kutta 4 method
    x = np.arange(x0, xf + h, h)
    n = len(x)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = rk4_step(fun, x[i], y[i], h)
    return x, y

t0 = 0
tf = 100
# 定义初始条件
y0 = [0.0, 0.0, 0.0, 0.0]
h = 0.001

x, y = rk4_solve(func, t0, tf, y0, h)

print(y[-1])