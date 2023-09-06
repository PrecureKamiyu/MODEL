import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# x轴方向上的长度344.5（从炉前一直到10温区左侧
# SOR迭代解泊松方程
# 网格为M*N   h = 0.1

h = 0.2
M = round(344.5 / h)

# 设l长度为5,遍历搜索
l = 10
N = round((2 * l) / h)

V = 25 * np.ones((M + 1, N + 1))

for i in range(1, M):
    if i <= round(25 / h):
        V[i, 0] = 6 * (h * i) + 25
        V[i, N] = 6 * (h * i) + 25       
    elif i <= round(197.5 / h):
        V[i, 0] = 175
        V[i, N] = 175    
    elif i <= round(202.5 / h):
        V[i, 0] = 4 * (h * i - 197.5) + 175
        V[i, N] = 4 * (h * i - 197.5) + 175       
    elif i <= round(233 / h):
        V[i, 0] = 195
        V[i, N] = 195       
    elif i <= round(238 / h):
        V[i, 0] = 8 * (h * i - 233) + 195
        V[i, N] = 8 * (h * i - 233) + 195        
    elif i <= round(268.5 / h):
        V[i, 0] = 235
        V[i, N] = 235     
    elif i <= round(273.5 / h):
        V[i, 0] = 4 * (h * i - 268.5) + 235
        V[i, N] = 4 * (h * i - 268.5) + 235
    elif i <= round(339.5 / h):
        V[i, 0] = 255
        V[i, N] = 255       
    elif i <= round(344.5 / h):
        V[i, 0] = 255 - 46 * (h * i - 339.5)
        V[i, N] = 255 - 46 * (h * i - 339.5)

# 至此初步建立了边界初值  接下来对内点进行SOR迭代：

# loop_times = 0
# beta = 1.5
# for i in range(0, loop_times):
#     print(i)
#     for j in range(1, N):
#         for k in range(1, M):
#             U[k, j] = beta * (U[k-1, j] + U[k+1, j] + U[k, j-1] + U[k, j+1]) / 4 + (1 - beta) * U[k, j]


# mnew = range(0, M+1)
# nnew = range(0, N+1)
# X, Y = np.meshgrid(nnew, mnew)
# Z = np.array(U)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)
# plt.show()

# 返回中心线处的温度
central_V = []
for i in range(0, M+1):
    central_V.append(V[i, 0])

for i in range(0, round((435.5 - 344.5)/h)):
    central_V.append(25)

