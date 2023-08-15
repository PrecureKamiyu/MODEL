# 二维插值

import numpy as np
from scipy import interpolate
import pylab as pl
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# e.g1:使函数图像清晰    （二维）
def func(x, y):
    return (x+y)*np.exp(-5.0*(x**2+y**2))

# x-y轴分为15*15的网格
y, x = np.mgrid[-1:1:15j, -1:1:15j]
# 计算每个网格点的函数值
fvals = func(x, y)
# 三次样条二维插值
newfunc1 = interpolate.interp2d(x, y, fvals, kind='cubic')
# 计算100*100网格上插值
xnew = np.linspace(-1, 1, 100)
ynew = np.linspace(-1, 1, 100)
fnew = newfunc1(xnew, ynew)

# 可视化
pl.subplot(121)
im1 = pl.imshow(fvals, extent=[-1, 1, -1, 1], cmap = mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im1)
pl.subplot(122)
im2 = pl.imshow(fnew, extent=[-1, 1, -1, 1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
pl.colorbar(im2)
pl.show()


# 三维图像的二维插值


# x-y轴分为20*20的网格
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
x, y = np.meshgrid(x, y)    # 网格点坐标矩阵
fvals = func(x, y)
# 划分图1
fig = plt.figure(figsize=(9, 6))
ax = plt.subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(x, y, fvals, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.colorbar(surf, shrink=0.5, aspect=5) # 颜色条标注

# 二维插值
newfunc2 = interpolate.interp2d(x, y, fvals, kind='cubic')
# 计算100*100网格上插值
xnew = np.linspace(-1, 1, 100)
ynew = np.linspace(-1, 1, 100)
fnew = newfunc2(xnew, ynew)
xnew, ynew = np.meshgrid(xnew, ynew)
ax2 = plt.subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax2.set_xlabel("xnew")
ax2.set_ylabel("ynew")
ax2.set_zlabel("fnew(x, y)")
plt.colorbar(surf2, shrink=0.5, aspect=5)
plt.show()
