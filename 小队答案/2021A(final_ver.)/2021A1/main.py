import pandas as pd
import numpy as np

data = pd.read_csv('data1.csv', usecols=range(1, 4),
                   encoding="gbk", dtype=float)

# 新建一个存放b所对应的促动器变化值总和的字典
b_dict = {}

# b在[-301, -299.8]的范围之间变动，把0.001作为变化步长进行遍历：
# -301, -299.8, 0.01
# -301, -299.8, 0.001
# -300.95, -300.85, 0.0001
for b in np.arange(-301, -299.8, 0.001):

    # 针对第一问的条件，由 f + CP = |b| 可得a关于b的函数关系
    a = 1/(4*(-b - 0.534*300.4))

    """
    在yOz平面中研究促动器伸缩量，取z = a*y^2 + b平面进行研究
    通过观察，发现理想抛物面上的点到原点C的距离d满足从顶点A0'到z = -1/2a的点先变小，
    再到工作态抛物面边界的点(满足sqrt(x^2 + y^2) = 150)又变大的单调性关系
    因此确定筛选条件：1. sqrt(z^2 + y^2) - R >= -0.6   ,  z = -1/2a
                    2. sqrt(z^2 + y^2) - R <= +0.6   ,  y = 150
    """
    if (-b/a - 1/(4*a**2))**0.5 - 300.4 < -0.6 or (150**2 + (a*150**2+b)**2)**0.5 - 300.4 > 0.6:
        continue
    else:
        b_dict[b] = 0
        for row in data.iloc:
            x0 = row[0]
            y0 = row[1]
            z0 = row[2]
            if x0**2 + y0**2 <= 22591:   # 根据极端情况  筛掉无用的点
                # 计算新的主索节点的纵坐标z1
                """
                先转化为求平面上的交点的问题
                联立：1. z = a*y^2 + b
                     2. z = z0/sqrt(x0^2 + y0^2) * y
                解得关于z的一元二次方程 Az^2 + Bz + C = 0
                """
                A = a*((x0*x0+y0*y0)/(z0*z0))
                B = -1
                C = b
                if A != 0:
                    # 所求的点在xOy平面以下，满足z<0，故取较小的z1
                    z1 = (-B-(B*B-4*A*C)**0.5)/(2*A)
                    y1 = (z1/z0) * y0
                    x1 = (z1/z0) * x0
                    # 根据相似关系 d = R - R(z1/z0)
                    d = 300.4*(1-z1/z0)
                    # 仅取工作态范围内的点
                    if x1 ** 2 + y1 ** 2 <= 22500:
                        b_dict[b] += abs(d)
                else:
                    # 最终加上顶点的变化量(因为顶点的xy坐标均为0，需要单独计算)
                    b_dict[b] += abs(300.4 + b)

# final_b为最终确定的最优参数
final_b = 0
min_sum = np.inf
# 遍历字典求得最小变化量所对应的b
for (key, value) in b_dict.items():
    if value < min_sum:
        min_sum = value
        final_b = key
final_a = 1/(4*(-final_b - 0.534*300.4))

print(f"z = {round(final_a,7)}(x^2+y^2)+{round(final_b,4)}")
