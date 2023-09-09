import numpy as np
import pylab as plt
import random

# yR in (-350, 0)
def return_dot_list(yR, delta_r, a):
    r = 108
    dot_list = []
    # 逐次增加半径
    while r <= 350 + yR:
        N = int(np.floor((2 * np.pi * r) / (a + 5)))
        delta_theta = 2 * np.pi / N        
        # 获得一圈点
        off =  delta_theta* random.uniform(0, 1)
        for i in range(0, N):
            dot_list.append((r * np.cos(delta_theta * i + off), r * np.sin(delta_theta * i + off) + yR))
        r += delta_r

    while r <= 350 - yR:
        N = int(np.floor((2 * np.pi * r) / (a + 5)))
        delta_theta = 2 * np.pi / N
        # 获得一圈点
        off =  delta_theta* random.uniform(0, 1)
        for i in range(0, N):
            x = r * np.cos(delta_theta * i +  off)
            y = r * np.sin(delta_theta * i +  off) + yR
            if x**2 + y** 2 <= 350**2:
                dot_list.append((x, y))
        r += delta_r
    return dot_list

list = return_dot_list(-200, 13.484, 6)

x = []
y = []
for i in list:
    x.append(i[0])
    y.append(i[1])

plt.scatter(x, y)
plt.show()