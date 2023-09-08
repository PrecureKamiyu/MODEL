import numpy as np
import pandas as pd
import pylab as plt
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

for i in range(0, len(x0)):
    plt.plot(x0[i], y0[i], '.', color= 'black')

plt.show()
