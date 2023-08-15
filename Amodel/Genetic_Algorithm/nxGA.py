import numpy as np
from sko.GA import GA

# 基于遗传算法的最优条件搜索

def function2(args):
    x, y = args
    return x ** 2 + y ** 2

# 定义约束
def cons1(x):
    return [x[0]+x[1]-10]
cons=cons1



def fun(solution):
    return function2(solution[:2])

# 遗传算法的最优指的是取到最小值
# n_dim变量数  prob_mut变异概率  size_pop种群组数量  max_iter迭代次数   ub lb上下限
ga = GA(func=fun, n_dim=2,    prob_mut=0.01,    size_pop=100,   max_iter=5000, lb=[0, 0], ub=[10, 10], constraint_eq=[cons])

# ga.run()方法得到最优解
best_x, best_z = ga.run()
print('best_x,y:', best_x, '\n', 'best_z:', best_z)
