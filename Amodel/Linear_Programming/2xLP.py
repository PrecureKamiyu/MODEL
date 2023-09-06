# 线性规划问题

from scipy import optimize
import numpy as np

# 2a - b

C = np.array([2, -1])


# a + b < 4
# a - b < 2


A = np.array([[1,  1],
              [1, -1]])
B = np.array([4, 2])

# 此处没有额外需要满足的等式条件

# Aeq = np.array([[0, 0]])   
# Beq = np.array([0])


# 定义参数范围
x_bound = (0, None)


# 求解规划函数的最小值，省略Aeq Beq
res = optimize.linprog(C, A, B, bounds=(x_bound, x_bound))


# 输出最小值和使得函数取得最小值的条件
print("fun = " + str(round(res.fun, 4)))
print("x = " + str(res.x))
