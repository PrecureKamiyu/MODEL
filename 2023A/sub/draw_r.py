import numpy as np
import pandas as pd
import random
import openpyxl as op
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


h_collect = 8
d = 4
D_O = 7
mid = 80
h = mid + h_collect / 2
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

print(special)

rlist = []
for i in special:
    rlist.append
    ((x0[i]**2 + y0[i]**2)**0.5)

# 半径分类
for i in range(0, len(rlist)-1):
    print(rlist[i+1] - rlist[i])
