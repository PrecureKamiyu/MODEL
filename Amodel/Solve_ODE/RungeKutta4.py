import numpy as np
import matplotlib.pyplot as plt

def fun(x, y):
    # Define the system of differential equations
    y1, y2 = y
    dy1 = y2
    dy2 = -y1
    return np.array([dy1, dy2])

def rk4_step(fun, x, y, h):
    # Compute one step of the Runge-Kutta 4 method
    k1 = h * fun(x, y)
    k2 = h * fun(x + h / 2, y + k1 / 2)
    k3 = h * fun(x + h / 2, y + k2 / 2)
    k4 = h * fun(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def rk4_solve(fun, x0, xf, y0, h):
    # Solve the system of differential equations using the Runge-Kutta 4 method
    x = np.arange(x0, xf + h, h)
    n = len(x)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = rk4_step(fun, x[i], y[i], h)
    return x, y

# Define the initial conditions and parameters
x0 = 0
xf = 9.4248
y0 = [0, 1]
h = 0.0001

# Solve the system of differential equations using the Runge-Kutta 4 method
x, y = rk4_solve(fun, x0, xf, y0, h)

# Print the results
print('The solution to the system of differential equations is:')
print(y[-1])