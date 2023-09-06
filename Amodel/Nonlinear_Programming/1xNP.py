# 非线性规划
# 单变量：

from scipy.optimize import minimize
import numpy as np

# 定义函数 a/x + x
def fun1(args):
    a = args
    v = lambda x :  a/x[0] + x[0]
    return v

# 给定的参数必须是元组
args = (1)
x0 = np.asarray((2))    # 初始猜测值


res = minimize(fun1(args), x0, bounds=[(0, None)],method='SLSQP')

print("success : " + str(res.success))
print("fun = " + str(res.fun))
print("x = " + str(res.x))