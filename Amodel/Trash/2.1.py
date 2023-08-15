# 一维插值（经过样本点）

import numpy as np
import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt

# 线性插值（line）和样条插值（splev）

x = [250, 275, 300, 325, 350]
y = [2.07, 5.85, 14.97, 19.68, 36.80]

# 开始插值
x_new = np.linspace(250, 350, 1000)  # 新的精确度

# 线性插值
f_liner = interpolate.interp1d(x, y)

# 样条插值
tck = interpolate.splrep(x, y)
y_bspline = interpolate.splev(x_new, tck)

# 可视化
plt.xlabel(u't')
plt.ylabel(u'%')
plt.plot(x, y, "o", label=u"原始数据")
plt.plot(x_new, f_liner(x_new), label=u"线性插值")
plt.plot(x_new, y_bspline, label=u"B-spline插值")
pl.legend()
pl.show()
