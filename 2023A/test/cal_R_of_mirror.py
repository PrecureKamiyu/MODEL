import numpy as np
import pandas as pd


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

mid = h - h_collect / 2




# 建立一个面板字典，key为面板编号，存储  从中心到反射点的向量  信息




check = 3

def check_and_return_front(check):
    for i in range(1, len(special)):
        if check < special[i] and i == 1:
            return []
        elif check < special[i]:
            return range(special[i - 2], special[i - 1])