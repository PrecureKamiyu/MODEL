import numpy as np
from sko.GA import GA

# 基于遗传算法的最优条件搜索

def function1(x):
    return 1/x + x


def fun(solution):
    return function1(solution[0])

# 遗传算法的最优指的是取到最小值
#       func>>>>, n_dim变量数  prob_mut变异概率  size_pop种群组数量   max_iter迭代次数   ub lb上下限
ga = GA(func=fun, n_dim=1,    prob_mut=0.01,    size_pop=100,       max_iter=20, lb=[0], ub=[100])

# ga.run()方法得到最优解
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
