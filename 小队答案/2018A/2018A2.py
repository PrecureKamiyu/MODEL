import numpy as np
import pylab as plt
from scipy import sparse


def T(d2):
    T0 = 65 + 273.15
    T5 = 37 + 273.15

    d1 = 0.6
    d3 = 3.6
    d4 = 5

    x1 = d1
    x2 = d1 + d2
    x3 = d1 + d2 + d3
    x4 = d1 + d2 + d3 + d4

    # 记录由步长确定的点数

    h = 0.1
    M = round(x4 / h)  + 1
    N = 3601

    x0 = 0; xL = 15.2 * 0.001; dx = h * 0.001
    t0 = 0; tF = 5400; dt = 1
    ks = 8.36; ke = 120
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

    m0 = (ke + k1 / dx) * np.ones(1)
    main_diag1 = (1 + 2 * r1) * np.ones(round(d1 / h) - 1)
    m1 = (k1 + k2) / dx * np.ones(1)
    main_diag2 = (1 + 2 * r2) * np.ones(round(d2 / h) - 1)
    m2 = (k2 + k3) / dx * np.ones(1)
    main_diag3 = (1 + 2 * r3) * np.ones(round(d3 / h) - 1)
    m3 = (k3 + k4) / dx * np.ones(1)
    main_diag4 = (1 + 2 * r4) * np.ones(round(d4 / h) - 1)
    m4 = (k4 / dx + ks) * np.ones(1)

    main_diag = np.concatenate((m0, main_diag1, m1, main_diag2, m2, main_diag3, m3, main_diag4, m4), axis = 0)

    op0 = - k1 / dx * np.ones(1)
    off_p1 = -r1 * np.ones(round(d1 / h) - 1)
    op1 = - k2 / dx * np.ones(1)
    off_p2 = -r2 * np.ones(round(d2 / h) - 1)
    op2 = - k3 / dx * np.ones(1)
    off_p3 = -r3 * np.ones(round(d3 / h) - 1)
    op3 = - k4 / dx * np.ones(1)
    off_p4 = -r4 * np.ones(round(d4 / h) - 1)

    off_diagp = np.concatenate((op0, off_p1, op1, off_p2, op2, off_p3, op3, off_p4),axis = 0)

    off_n1 = -r1 * np.ones(round(d1 / h) - 1)
    on1 = - k1 / dx * np.ones(1)
    off_n2 = -r2 * np.ones(round(d2 / h) - 1)
    on2 = - k2 / dx * np.ones(1)
    off_n3 = -r3 * np.ones(round(d3 / h) - 1)
    on3 = - k3 / dx * np.ones(1)
    off_n4 = -r4 * np.ones(round(d4 / h) - 1)
    on4 = - k4 / dx * np.ones(1)

    off_diagn = np.concatenate((off_n1, on1, off_n2, on2, off_n3, on3, off_n4, on4),axis = 0)

    diagonals = [main_diag, off_diagp, off_diagn]

    l = round(x4 / h) + 1

    A = sparse.diags(diagonals, [0, 1, -1], shape=(l, l)).toarray()

    T = np.zeros((M, N))


    T[:, 0] = T5 + 0 * xspan


    for k in range(1, N):
        print(k)
        c = np.zeros(M-2)
        b1 = np.array([ke * T0, ks * T5])
        b1 = np.insert(b1, 1, c)
        b21 = T[1:round(x1 / h), k-1]
        b22 = T[round(x1 / h)+1 : round(x2 / h), k-1]
        b23 = T[round(x2 / h)+1 : round(x3 / h), k-1]
        b24 = T[round(x3 / h)+1 : M-1, k-1]
        b2 = np.hstack(([0], b21, [0], b22, [0], b23, [0], b24, [0]))
        b = b1 + b2
        T[0:M, k] = np.linalg.solve(A, b)

    return [T[M-1, 3600], T[M-1, 3300]]

def binarySearch(d_list, l, r):
    if r > l:
        mid = int((r + l) / 2)
        T_list = T(d_list[mid])

        if T_list[0] < (47+273.15) and T_list[1] < (44+273.15):
            return binarySearch(d_list, l, mid)
        else:
            return binarySearch(d_list, mid + 1, r)
    else:
        return d_list[l]
    
d_list = np.arange(0.6, 25.01, 0.01)

best_d2 = binarySearch(d_list, 0, len(d_list)-1)
print(best_d2)
