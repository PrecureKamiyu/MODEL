# 一维插值（经过样本点）
# 别再用np.polyfit了！
# 把题目所给的散点数据连续化

import numpy as np
from scipy import interpolate
import pylab as plt

x = [50, 75, 100, 125, 150]
y = [0, 100, 0, -25, 0]

# 确定新的精确度
x_new = np.linspace(50, 150, 1000)

# 样条插值
tck = interpolate.splrep(x, y)
y_new = interpolate.splev(x_new, tck)

# 可视化
plt.rc('text', usetex= True)
plt.rc('font', size= 15)
plt.xlabel('$x$')
plt.ylabel('$y$', rotation = 0)
plt.plot(x, y, 'o')  # 显示原始点数据
plt.plot(x_new, y_new, label="Bspline_IP")  # 显示样条插值

# 存在多条曲线时，legend会在右上角生成对应颜色的标签
plt.legend()
plt.show()
