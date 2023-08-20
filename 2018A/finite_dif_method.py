import numpy as np
import pylab as plt
from scipy import sparse

# 记录点数
M = 50; N = 60

# 空间时间步长
x0 = 0; xL = 1; dx = (xL - x0)/(M - 1)
t0 = 0; tF = 0.2; dt = (tF - t0)/(N - 1)

# 参数
D = 0.1; alpha = -3

r = dt * D/ dx ** 2
s = dt * alpha

xspan = np.linspace(x0, xL, M)
tspan = np.linspace(t0, tF, N)

main_diag = (1 + 2 * r - s) * np.ones(M-2)
off_diag  = -r * np.ones(M-3)

n = M - 2
diagonals = [main_diag, off_diag, off_diag]

A = sparse.diags(diagonals, [0, -1, 1], shape=(n, n)).toarray()

U = np.zeros((M, N))
U[:,0] = 4 * xspan - 4 * xspan ** 2
U[0,:] = 0.0
U[-1,:] = 0.0

for k in range(1, N):
    c = np.zeros(M-4)
    b1 = np.array([r*U[0, k], r*U[-1, k]])
    b1 = np.insert(b1, 1, c)
    b2 = U[1:M-1, k-1]
    b = b1 + b2
    U[1:M-1, k] = np.linalg.solve(A, b)

plt.rc('text', usetex= True)
plt.rc('font', size = 15)
X, T = np.meshgrid(tspan, xspan)

ax = plt.axes(projection = '3d')
ax.plot_surface(X, T, U, linewidth = 0, cmap = plt.cm.coolwarm)
ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
ax.set_xlabel('$t$')
ax.set_ylabel('$s$')
ax.set_zlabel('$u$')
plt.tight_layout()
plt.show()

print(xspan)
print(tspan)
