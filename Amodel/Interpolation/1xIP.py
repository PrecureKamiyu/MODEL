# 一维插值（经过样本点）

import numpy as np
import pylab as pl
from scipy import interpolate
import pylab as plt

x = [50, 75, 100, 125, 150]
y = [0, 25, 0, -25, 0]

# 确定新的精确度
x_new = np.linspace(50, 150, 1000)

# 线性插值
f_liner = interpolate.interp1d(x, y)

# 样条插值
tck = interpolate.splrep(x, y)
y_bspline = interpolate.splev(x_new, tck)

# 可视化
plt.rc('text', usetex= True)
plt.rc('font', size= 15)
plt.xlabel('$x$')
plt.ylabel('$y$', rotation = 0)
plt.plot(x, y, 'o')  # 显示原始点数据
plt.plot(x_new, f_liner(x_new), label="linear_IP")  # 显示线性插值
plt.plot(x_new, y_bspline, label="Bspline_IP")  # 显示样条插值
pl.legend()
pl.show()
