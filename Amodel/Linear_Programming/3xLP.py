# 线性规划问题

from scipy import optimize
import numpy as np


# 决策变量为 x = [a, b, c]^T
# 确定规划函数 2a + 3b - 5c
C = np.array([2, 3, -5])


# 确定约束条件（默认小于号）
# A @ x < B
# -2a + 5b - c < -10
#   a + 3b + c <  12


A = np.array([[-2, 5, -1],
              [ 1, 3,  1]])
B = np.array([-10, 12])


# 需要满足的等式条件(特别说明：A、Aeq矩阵都应该是二维的矩阵)
# Aeq @ x = Beq
# a + b + c = 7
Aeq = np.array([[1, 1, 1]])   
Beq = np.array([7])


# 定义每个自变量范围
x_bound = (0, None)


# 求解规划函数的最小值
res = optimize.linprog(C, A, B, Aeq, Beq, bounds=(x_bound, x_bound, x_bound))

# 输出最小值和使得函数取得最小值的条件
print("fun = " + str(round(res.fun,4)))
print("x = " + str(res.x))



# 求解规划函数的最大值

res = optimize.linprog(-C, A, B, Aeq, Beq, bounds=(x_bound, x_bound, x_bound))

# 此处的解加上负号获得最大值
print("fun = " + str(round(-res.fun,4)))
print("x = " + str(res.x))
