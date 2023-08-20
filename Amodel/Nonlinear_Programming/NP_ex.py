from scipy.optimize import minimize
import numpy as np


def fun(x):
    v = -(2 * x[0] + 3 * x[0] ** 2 + 3 * x[1] + x[1]**2 + x[2])
    return v

def con():
    cons = ({'type':'ineq', 'fun':lambda x : 10 - (x[0] + 2 * x[0] ** 2 + x[1] + 2 * x[1]**2 + x[2])},
            {'type':'ineq', 'fun':lambda x : 50 - (x[0] +  x[0] ** 2 + x[1] + x[1]**2 - x[2])},
            {'type':'ineq', 'fun':lambda x : 40 - (2 * x[0] + x[0] ** 2 + 2 * x[1] + x[2])},
            {'type':'ineq', 'fun':lambda x : x[0] + 2 * x[1] - 1},
            {'type':  'eq', 'fun':lambda x : x[0] ** 2 + x[2] - 2})
    return cons

cons = con()

x0 = np.array((1, 1, 1))

res = minimize(fun, x0, method='SLSQP',bounds=[(0,None),(None,None),(None,None)], constraints=cons)

print(res.success)
print(res.x)
print(res.fun)