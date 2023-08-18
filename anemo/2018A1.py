import numpy as np
import pylab as plt
from scipy import sparse
import pandas as pd

T0 = 75 + 273.15
T5 = 37 + 273.15

data = pd.read_excel('CUMCM-2018-Problem-A-Chinese-Appendix.xlsx', sheet_name='附件2')
first_column = data.iloc[1:, 1]


def tpr(k, t):
    return T0 - (T0-T5)* np.exp(-k * t)



# 记录由步长确定的点数

h = 0.01
M = int(15.2 / h)  + 1
N = 5401

x0 = 0; xL = 15.2; dx = h
t0 = 0; tF = 5400; dt = 1

k1 = 0.082;  k2 = 0.37;  k3 = 0.045;  k4 = 0.028
rho1 = 300; rho2 = 862; rho3 = 74.2; rho4 = 1.18
c1 =  1377;  c2 = 2100;   c3 = 1726;   c4 = 1005

a1 = k1 / (rho1 * c1)
a2 = k2 / (rho2 * c2)
a3 = k3 / (rho3 * c3)
a4 = k4 / (rho4 * c4)

r1 = dt * a1 / dx ** 2
r2 = dt * a2 / dx ** 2
r3 = dt * a3 / dx ** 2
r4 = dt * a4 / dx ** 2

xspan = np.linspace(x0, xL, M)
tspan = np.linspace(t0, tF, N)

main_diag1 = (1 + 2 * r1) * np.ones(59)
main_diag2 = (1 + 2 * r2) * np.ones(600)
main_diag3 = (1 + 2 * r3) * np.ones(360)
main_diag4 = (1 + 2 * r4) * np.ones(500)

main_diag = np.concatenate((main_diag1,main_diag2,main_diag3,main_diag4), axis = 0)

off_p1 = -r1 * np.ones(59)
off_p2 = -r2 * np.ones(600)
off_p3 = -r3 * np.ones(360)
off_p4 = -r4 * np.ones(499)

off_diagp = np.concatenate((off_p1,off_p2,off_p3,off_p4),axis = 0)

off_n1 = -r1 * np.ones(58)
off_n2 = -r2 * np.ones(600)
off_n3 = -r3 * np.ones(360)
off_n4 = -r4 * np.ones(500)

off_diagn = np.concatenate((off_n1,off_n2,off_n3,off_n4),axis = 0)

diagonals = [main_diag, off_diagp, off_diagn]

l = 59 + 600 + 360 + 500

A = sparse.diags(diagonals, [0, 1, -1], shape=(l, l)).toarray()

T = np.zeros((M, N))


T[:, 0] = T5 + 0 * xspan

k = 1

T0T = []
for t in range(0, 5401):
    T0T.append(tpr(k, t))
T[0, :] = np.array(T0T)


temp_list = []
for i in range(1, 5402):
    temp_list.append(first_column[i] + 273.15)
T[-1, :] = np.array(temp_list)


for k in range(1, N):
    print(k)
    c = np.zeros(M-4)
    b1 = np.array([r1*T[0, k], r4*T[-1, k]])
    b1 = np.insert(b1, 1, c)
    b2 = T[1:M-1, k-1]
    b = b1 + b2
    T[1:M-1, k] = np.linalg.solve(A, b)

plt.rc('text', usetex= True)
plt.rc('font', size = 15)
X, U = np.meshgrid(tspan, xspan)

ax = plt.axes(projection = '3d')
ax.plot_surface(U, X, T, linewidth = 0, cmap = plt.cm.coolwarm)
ax.set_xticks([0, 3,6,9,12,15])
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u$')
plt.tight_layout()
plt.show()

