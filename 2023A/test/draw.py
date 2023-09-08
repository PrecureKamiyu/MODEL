import numpy as np
import matplotlib.pyplot as plt
a = 5
b = 0.6
yR = 100
xi = (1 + np.sqrt(5)) / 2
klist = []
thetalist = []
xlist = []
ylist = []
for k in range(1, 2000):
    r = a * k ** b
    theta = 2 * np.pi * xi ** (-2) * k
    x = r * np.cos(theta)
    y = r * np.sin(theta) - yR
    if (y + yR)**2 + x**2 > 100**2 and x**2 + y**2 < 350**2:
        klist.append(k)
        thetalist.append(theta % (2 * np.pi))
        xlist.append(x)
        ylist.append(y)
    # 简化模型  以第一个点为邻域   取最近距离D1i为最短距离与a+5比较
plt.scatter(xlist, ylist)
print(thetalist)
plt.show()