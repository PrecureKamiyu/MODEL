import numpy as np
import pandas as pd
import random

"""
STEP1
"""

data = pd.read_excel('附件.xlsx', sheet_name= 'Sheet1')

# 中心坐标
x0 = []
y0 = []
for row in data.iloc:
    x0.append(row[0])
    y0.append(row[1])

h = 80
h_collect = 8
d = 4
D_O = 7
mid = h - h_collect / 2
xR = 0
yR = 0
a = 6
b = 6
eta_ref = 0.92
# 建立一个面板字典，key为面板编号，存储  从中心到反射点的向量  信息

mirror = {}
for i in range(0, len(x0)):
    n = [x0[i] - xR, y0[i] - yR, d - mid]
    n = n / np.linalg.norm(n)
    mirror[i] = n

# 根据半径分类第几排
mirror_R = {}

for i in range(0, len(x0)):

    mirror_R[i] = round((x0[i]**2 + y0[i]**2)**0.5)

# 获得special间断点列表
special = [0]
front = 108
for i in range(0, len(x0)):
    if mirror_R[i] == front:
        continue
    else:
        front = mirror_R[i]
        special.append(i)
special.append(len(x0))

# 判断并返回前排列表函数
def check_and_return_front(check):
    for i in range(1, len(special)):
        if check < special[i] and i == 1:
            return []
        elif check < special[i]:
            return range(special[i - 2], special[i - 1])


Day_list = [305, 336, 0, 30, 60, 91, 121, 152, 183, 213, 244, 274]
ST_list = [9, 10.5, 12, 13.5, 15]

# 以下写某两组个循环
# 先以3.21的9点为例

for i in Day_list:
    for j in ST_list:
        D = i
        ST = j

        phi = np.deg2rad(39.4)
        omega = np.pi / 12 * (ST - 12)
        sindelta = np.sin(2 * np.pi * D / 365) * np.sin(2 * np.pi * 23.45 / 360)

        cosdelta = (1 - sindelta**2)**0.5
        # alphas
        sinalphas = cosdelta * np.cos(phi) * np.cos(omega) + sindelta * np.sin(phi)
        cosalphas = (1 - sinalphas**2)**0.5
        # gammas
        cosgammas = (sindelta - sinalphas * np.sin(phi))/(cosalphas * np.cos(phi))
        singammas = np.sign(ST - 12) * (1 - cosgammas**2)**0.5

        # 确定所有面板的法向量

        print(D, ST, sinalphas, cosgammas)
