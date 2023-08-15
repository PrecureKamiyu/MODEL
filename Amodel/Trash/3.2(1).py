# 一阶
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义微分方程
def model(y, x):
    dydx = -2 * y
    return dydx

# 定义初始条件
y0 = 1

# 定义 x 范围
x = np.linspace(0, 5, 100)

# 求解微分方程
y = odeint(model, y0, x)

# 绘制数值解
plt.plot(x, y, label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution of dy/dx = -2y')
plt.legend()
plt.grid()
plt.show()

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义微分方程
def model(y, x):
    dydx = -2 * y
    return dydx

# 定义初始条件
y0 = 1

# 定义 x 范围
x = np.linspace(0, 5, 100)

# 求解微分方程
y = odeint(model, y0, x)
# 获取特定 x 值对应的 y 值
x_specific = 2.5
index = np.abs(x - x_specific).argmin()
y_specific = y[index]
print(f"At x = {x_specific}, y = {y_specific}")

# 绘制数值解
plt.plot(x, y, label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution of dy/dx = -2y')
plt.legend()
plt.grid()
plt.show()
