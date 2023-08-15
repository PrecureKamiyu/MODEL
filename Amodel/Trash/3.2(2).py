# 二阶

# d^2x/dt^2 + 2 * dx/dt + 2 * y = 0
# d^2y/dt^2 + y = 0
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义微分方程组
def model(u, t):
    x, u, y, v = u[0], u[1], u[2], u[3]
    dxdt = u
    dudt = -2 * u - 2 * y
    dydt = v
    dvdt = -y
    return [dxdt, dudt, dydt, dvdt]

# 定义初始条件
x0 = 1
u0 = 0  # 初始 dx/dt = 0
y0 = 1
v0 = 0  # 初始 dy/dt = 0
u_init = [x0, u0, y0, v0]

# 定义时间范围
t = np.linspace(0, 10, 100)

# 求解微分方程组
u = odeint(model, u_init, t)
x = u[:, 0]
y = u[:, 2]


# 获取特定位置的解值
t_specific = 5.0  # 想要获取的特定时间
index = np.abs(t - t_specific).argmin()
x_specific = x[index]
y_specific = y[index]
print(f"At t = {t_specific}, x = {x_specific}, y = {y_specific}")

# 绘制数值解
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.xlabel('t')
plt.ylabel('x(t), y(t)')
plt.title('Numerical Solution of a System of Second Order Differential Equations')
plt.legend()
plt.grid()
plt.show()
