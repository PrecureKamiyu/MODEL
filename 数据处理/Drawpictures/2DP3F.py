import pylab as plt
import matplotlib as mpl
import numpy as np

def func(x, y):
    return  x**2 + y**2

# 此处是 mgrid   直接获得网格化数据类型
Y, X = np.mgrid[-1:1:101j, -1:1:101j]

fvals = func(X, Y)

im1 = plt.imshow(fvals, extent=[-1, 1, -1, 1], cmap= mpl.cm.hot , origin= 'lower')
plt.colorbar(im1)
plt.show()