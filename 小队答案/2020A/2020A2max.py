import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def calculateCurve(T1, T6, T7, T8, v):
    h = 0.2
    M = round(344.5 / h)
    V = 25 * np.ones(M + 1)

    # 修改温区段落
    T0 = 25     # fixed
    T10 = 25    # fixed

    for i in range(1, M):
        if i <= round(25 / h):
            V[i] = ((T1 - T0) / 25) * (h * i) + T0
        elif i <= round(197.5 / h):
            V[i] = T1
        elif i <= round(202.5 / h):
            V[i] = ((T6 - T1) / 5) * (h * i - 197.5) + T1
        elif i <= round(233 / h):
            V[i] = T6
        elif i <= round(238 / h):
            V[i] = ((T7 - T6) / 5) * (h * i - 233) + T6
        elif i <= round(268.5 / h):
            V[i] = T7
        elif i <= round(273.5 / h):
            V[i] = ((T8 - T7) / 5) * (h * i - 268.5) + T7
        elif i <= round(339.5 / h):
            V[i] = T8
        elif i <= round(344.5 / h):
            V[i] = T8 - ((T8 - T10) / 5) * (h * i - 339.5)

    # 返回中心线处的温度
    central_V = []
    for i in range(0, M+1):
        central_V.append(V[i])

    for i in range(0, round((440 - 344.5)/h)):
        central_V.append(25)

    # 共参数
    dy = 0.001 * 0.001
    dt = 0.5
    l = 0.15 * 0.001
    M = round(l / dy)

    """
    I阶段
    """
    a = 5.0 * 10 ** -11
    H = 5.0 * 10 ** -6
    t = round(200 / v)
    N = round(t / dt)

    r = dt * a / dy ** 2

    # 获得系数矩阵A
    m0 = (1 - (H / (H + dy) - 2) * r) * np.ones(1)
    main_diag0 = (1 + 2 * r) * np.ones(M - 3)
    main_diag = np.concatenate((m0, main_diag0, m0), axis=0)
    off_diag = -r * np.ones(M-2)
    diagonals = [main_diag, off_diag, off_diag]
    l = M-1
    A = sparse.diags(diagonals, [0, 1, -1], shape=(l, l)).toarray()

    # 获得炉温中心与时间、位置的关系U1
    U = np.zeros((M, N))
    U[:, 0] = 25
    for k in range(1, N):
        c = np.zeros(M-3)
        b1 = np.array([(dy / (dy + H)) * r * central_V[round(k*dt*v/h)],
                       (dy / (dy + H)) * r * central_V[round(k*dt*v/h)]])
        b1 = np.insert(b1, 1, c)
        b2 = U[1:M, k-1]
        b = b1 + b2
        U[1:M, k] = np.linalg.solve(A, b)

    """
    II阶段
    """
    a = 6.0 * 10 ** -11
    H = 1.0 * 10 ** -7
    t2 = round(342 / v)
    N2 = round((t2 - t) / dt)

    r = dt * a / dy ** 2

    # 获得系数矩阵A
    m0 = (1 - (H / (H + dy) - 2) * r) * np.ones(1)
    main_diag0 = (1 + 2 * r) * np.ones(M - 3)
    main_diag = np.concatenate((m0, main_diag0, m0), axis=0)
    off_diag = -r * np.ones(M-2)
    diagonals = [main_diag, off_diag, off_diag]
    l = M-1
    A = sparse.diags(diagonals, [0, 1, -1], shape=(l, l)).toarray()

    # 获得炉温中心与时间、位置的关系U2
    U2 = np.zeros((M, N2))
    U2[:, 0] = U[:, N-1]
    for k in range(1, N2):
        c = np.zeros(M-3)
        b1 = np.array([(dy / (dy + H)) * r * central_V[round((k*dt*v + 200)/h)],
                       (dy / (dy + H)) * r * central_V[round((k*dt*v + 200)/h)]])
        b1 = np.insert(b1, 1, c)
        b2 = U2[1:M, k-1]
        b = b1 + b2
        U2[1:M, k] = np.linalg.solve(A, b)

    """
    III阶段
    """
    a = 3.0 * 10 ** -11
    H = 1.0 * 10 ** -5
    t3 = round(435.5 / v)
    N3 = round((t3 - t2) / dt)

    r = dt * a / dy ** 2

    # 获得系数矩阵A
    m0 = (1 - (H / (H + dy) - 2) * r) * np.ones(1)
    main_diag0 = (1 + 2 * r) * np.ones(M - 3)
    main_diag = np.concatenate((m0, main_diag0, m0), axis=0)
    off_diag = -r * np.ones(M-2)
    diagonals = [main_diag, off_diag, off_diag]
    l = M-1
    A = sparse.diags(diagonals, [0, 1, -1], shape=(l, l)).toarray()

    # 获得炉温中心与时间、位置的关系U3
    U3 = np.zeros((M, N3))
    U3[:, 0] = U2[:, N2-1]
    for k in range(1, N3):
        c = np.zeros(M-3)
        b1 = np.array([(dy / (dy + H)) * r * central_V[round((k*dt*v + 342)/h)],
                       (dy / (dy + H)) * r * central_V[round((k*dt*v + 342)/h)]])
        b1 = np.insert(b1, 1, c)
        b2 = U3[1:M, k-1]
        b = b1 + b2
        U3[1:M, k] = np.linalg.solve(A, b)

    # 获得阶段温度曲线
    step1 = []
    for k in range(0, N):
        step1.append(U[round(M/2), k])
    step2 = []
    for k in range(0, N2):
        step2.append(U2[round(M/2), k])
    step3 = []
    for k in range(0, N3):
        step3.append(U3[round(M/2), k])

    tspan1 = np.arange(0, t, 0.5)
    tspan2 = np.arange(t, t2, 0.5)
    tspan3 = np.arange(t2, t3, 0.5)

    tspan = np.concatenate((tspan1, tspan2, tspan3), axis=0)

    # 最终要将step写入到result.csv中
    step = np.concatenate((step1, step2, step3), axis=0)

    # plt.plot(tspan, step)
    # plt.show()
    return [tspan, step]


# 焊炉总长度
l0 = 435.5
y = []
for v in np.arange(65 / 60, 100 / 60, 0.1 / 60):
    print(v)
    [tspan, step] = calculateCurve(182, 203, 237, 254, v)
    # 拟合
    coefficients = np.polyfit(tspan, step, 10)
    # tspan_new = np.arange(0, 100, 1)
    # step_new = np.polyval(coefficients, tspan_new)
    tspan_new = np.arange(0, l0 / v, 0.01)
    y.append(max(np.polyval(coefficients, tspan_new)))

plt.plot(np.arange(65 / 60, 100 / 60, 0.1 / 60), y)
plt.show()
