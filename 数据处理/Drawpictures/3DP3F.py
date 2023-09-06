import numpy as np
import pylab as plt

x = y = np.linspace(-2, 2, 101)

def func(x, y):
    return x**2 + y**2

X, Y = np.meshgrid(x, y)

Z = func(X, Y)


plt.rc('text', usetex= True)
plt.rc('font', size = 15)

# 创建一个图像
fig1 = plt.figure()
# 添加子图
ax = fig1.add_subplot(111, projection = '3d')
# 画三维曲面图
ax.plot_surface(X, Y, Z, cmap = plt.cm.coolwarm)

# 设置明晰的坐标位置，一定要注意单位  数量级
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.show()