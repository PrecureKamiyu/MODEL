# 二维插值

import numpy as np
from scipy import interpolate
import pylab as pl
import matplotlib as mpl

# 示例函数，用于获得跨度大的散点图
def func(x, y):
    return (x+y)*np.exp(-5.0*(x**2+y**2))

# xOy平面分为15*15的网格
y, x = np.mgrid[-1:1:15j, -1:1:15j]

# 获得原始数据
fvals = func(x, y)


# 二维插值函数
# kind = 'linear'线性插值  'cubic'三次样条插值  'quintic'五次样条插值
newfunc1 = interpolate.interp2d(x, y, fvals, kind='linear')
newfunc2 = interpolate.interp2d(x, y, fvals, kind='cubic')
newfunc3 = interpolate.interp2d(x, y, fvals, kind='quintic')



# 计算100*100网格上插值
xnew = np.linspace(-1, 1, 101)
ynew = np.linspace(-1, 1, 101)
f1new = newfunc1(xnew, ynew)
f2new = newfunc2(xnew, ynew)
f3new = newfunc3(xnew, ynew)

# 创建二行二列的多图，最后一个数字代表第几张子图

pl.subplot(221)
im1 = pl.imshow(fvals, extent=[-1, 1, -1, 1], cmap = mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im1)

pl.subplot(222)
im2 = pl.imshow(f1new, extent=[-1, 1, -1, 1], cmap = mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im2)

pl.subplot(223)
im3 = pl.imshow(f2new, extent=[-1, 1, -1, 1], cmap = mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im3)

pl.subplot(224)
im4 = pl.imshow(f3new, extent=[-1, 1, -1, 1], cmap = mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im4)

pl.show()