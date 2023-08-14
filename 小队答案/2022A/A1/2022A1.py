import math
from scipy.integrate import odeint
import numpy as np
import pylab as plt
import openpyxl as op

workbook = op.Workbook()
worksheet1 = workbook.active
worksheet1.title = "调整后主索节点编号及坐标"
data1 = [["时间 (s)","浮子位移 (m)","浮子速度(m/s)","振子位移(m)","振子速度(m/s)"]]


def func(z, t, f, w, k1, k2, M, m, p, g, R):
    x, dxdt, y, dydt = z
    d2xdt2 = (- k1 * (x - y) - k2 * (dxdt - dydt)) / m
    d2ydt2 = (f * np.cos(w * t) + k1 * (x - y) + k2 * (dxdt - dydt) - k3 * dydt - p * g * np.pi * (R ** 2) * y) / (M + M_prime)
    return [dxdt, d2xdt2, dydt, d2ydt2]

# 定义参数
f = 6250
w = 1.4005
k1 = 80000
k2 = 10000
k3 = 656.3616
M = 4866
m = 2433
M_prime = 1335.535
p = 1025
g = 9.8
R = 1

# 定义初始条件和时间范围
y0 = [0.0, 0.0, 0.0, 0.0]
# 
t = np.linspace(0, 180, 901)

# 求解方程组
sol = odeint(func, y0, t, args=(f, w, k1, k2, M, m, p, g, R))

for i in range(0, 901):
    data1.append([i*0.2,sol[i][0],sol[i][1],sol[i][2],sol[i][3]])

plt.plot(t, sol, '*')
plt.show()

print(sol[50])

for row in data1:
    worksheet1.append(row)
workbook.save(filename="result1-1.xlsx")