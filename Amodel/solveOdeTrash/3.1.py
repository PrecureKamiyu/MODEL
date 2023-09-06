# 微分方程

# 解析解

import numpy as np
import sympy


# 应用初始条件到解决方案上 initial conditions
def apply_ics(sol, ics, x, known_params):
    free_params = sol.free_symbols - set(known_params)
    eqs = [(sol.lhs.diff(x, n) - sol.rhs.diff(x, n)).subs(x, 0).subs(ics) for n in range(len(ics))]
    sol_params = sympy.solve(eqs, free_params)
    return sol.subs(sol_params)


# 将解析解以字符串形式打印
sympy.init_printing()   # 初始化打印环境
t, omega0, gamma = sympy.symbols("t, omega_0, gamma", positive=True)  # 标记参数且均为正
x = sympy.Function('x')  # x微分函数，不是变量(即x为关于t的函数x（t）)
ode = x(t).diff(t, 2) + 2*gamma*omega0*x(t).diff(t) + omega0**2*x(t)
ode_sol = sympy.dsolve(ode) # 用diff()和dsolve()得到通解
ics = {x(0):1, x(t).diff(t).subs(t, 0):0}  # 初始条件字典匹配
x_t_sol = apply_ics(ode_sol, ics, t, [omega0, gamma])
sympy.pprint(x_t_sol)
