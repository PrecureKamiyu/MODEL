# 数值解（算不出解析解）

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import sympy

# 画场线图
def plot_direction_field(x, y_x, f_xy, x_lim=(-5, 5), y_lim=(-5, 5), ax=None):
    f_np = sympy.lambdify((x, y_x), f_xy, 'numpy')
    x_vec = np.linspace(x_lim[0], x_lim[1], 20)
    y_vec = np.linspace(y_lim[0], y_lim[1], 20)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    dx = x_vec[1] - x_vec[0]
    dy = y_vec[1] - y_vec[0]
    for m, xx in enumerate(x_vec):
        for n, yy in enumerate(y_vec):
            Dy = f_np(xx, yy)
            Dx = 0.8 * dx ** 2 / np.sqrt(dx ** 2 + Dy ** 2)
            Dy = 0.8 * Dy * dy / np.sqrt(dx ** 2 + Dy ** 2)
            ax.plot([xx - Dx / 2, xx + Dx / 2], [yy - Dy / 2, yy + Dy / 2], 'b', lw=0.5)
    ax.axis('tight')
    ax.set_title(r"$%s$" % (sympy.latex(sympy.Eq(y_x.diff(x), f_xy))), fontsize=18)
    return ax

# 求数值解
x = sympy.symbols('x')
y = sympy.Function('y')
f = x-y(x)**2
f_np = sympy.lambdify((y(x), x), f) # 上一行符号表达式转成隐函数
y0 = 1
xp = np.linspace(0, 5, 100)
yp = integrate.odeint(f_np, y0, xp) # 初始y0解f_np，x范围xp
xn = np.linspace(0, -5, 100)
yn = integrate.odeint(f_np, y0, xn)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_direction_field(x, y(x), f, ax=ax)  # 绘制f的场线图
ax.plot(xn, yn, 'b', lw=2)
ax.plot(xp, yp, 'r', lw=2)
plt.show()
