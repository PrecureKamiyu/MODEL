from scipy.optimize import minimize
import numpy as np

# 待规划函数
def fun2(args):
    a, b, c, d = args
    v = lambda x : (a/x[0])/(b+x[1]) - c*x[0] + d*x[2]
    return v

# 约束条件    分为eq（=0）和ineq（>=0）

#  x0 * x1 <= a
#  x0 + x2 == b

def con(args):
    a, b = args
    cons = ({'type':'ineq', 'fun':lambda x : a - x[0] * x[1]},
            {'type':  'eq', 'fun':lambda x : x[0] + x[2] - b})
    return cons


# fun2所对应的参数
args = (2, 1, 3, 4)

# 约束条件对应的参数
args1 = (0.45, 1)
cons = con(args1)

# 初始猜测值
x0 = np.asarray((0.5, 0.5, 0.5))

# 给定三个变量的取值范围，求在满足约束条件下的待规划函数的最小值
res = minimize(fun2(args), x0, method='SLSQP', bounds=[(0.1,0.9),(0.1,0.9),(0.1,0.9)], constraints=cons)

print("success1 : " + str(res.success))
print("fun1 = " + str(res.fun))
print("x = " + str(res.x))