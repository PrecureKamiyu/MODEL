import numpy as np
from scipy import sparse
import csv
import matplotlib.pyplot as plt

h = 0.2
M = round(344.5 / h)

V = 25 * np.ones(M + 1)

for i in range(1, M):
    if i <= round(25 / h):
        V[i] = 6 * (h * i) + 25 
    elif i <= round(197.5 / h):
        V[i] = 175
    elif i <= round(202.5 / h):
        V[i] = 4 * (h * i - 197.5) + 175  
    elif i <= round(233 / h):
        V[i] = 195 
    elif i <= round(238 / h):
        V[i] = 8 * (h * i - 233) + 195 
    elif i <= round(268.5 / h):
        V[i] = 235
    elif i <= round(273.5 / h):
        V[i] = 4 * (h * i - 268.5) + 235
    elif i <= round(339.5 / h):
        V[i] = 255  
    elif i <= round(344.5 / h):
        V[i] = 255 - 46 * (h * i - 339.5)


# 返回中心线处的温度
central_V = []
for i in range(0, M+1):
    central_V.append(V[i])

for i in range(0, round((435.5 - 344.5)/h)):
    central_V.append(25)

# 获得标准趋近温度
perfect_U = []
with open("附件.csv", "r") as csvfile:
    reader = csv.reader(csvfile)  # 创建一个reader对象
    for row in reader:
        perfect_U.append(float(row[1]))


v = 70 / 60
dy = 0.0002 * 0.001
dt = 0.1

a = 4.9 * 10 ** -11
H = 4 * 10 ** -6

l = 0.15 * 0.001
t = 171  # 变动项
M = round(l / dy)
N = round(t / dt)   # 变动项

aHsdict = {}
for ax in np.linspace(4, 6, 21):
    print(ax)
    for Hx in np.linspace(3, 5, 10):
        a = ax * 10 ** -11
        H = Hx * 10 ** -6

        r = dt * a / dy ** 2 

        # 获得系数矩阵A
        m0 = (1 - (H / (H + dy) - 2) * r) * np.ones(1)
        main_diag0 = (1 + 2 * r) * np.ones(M - 3)
        main_diag = np.concatenate((m0, main_diag0, m0), axis = 0)
        off_diag = -r * np.ones(M-2)
        diagonals = [main_diag, off_diag, off_diag]
        l = M-1
        A = sparse.diags(diagonals, [0, 1, -1], shape=(l, l)).toarray()

        # 获得炉温中心与时间、位置的关系U
        U = np.zeros((M, N))
        U[:,0] = 25


        for k in range(1, N):
            c = np.zeros(M-3)
            b1 = np.array([(dy / (dy + H)) * r * central_V[round(k*dt*v/h)], (dy / (dy + H)) * r * central_V[round(k*dt*v/h)]])
            b1 = np.insert(b1, 1, c)
            b2 = U[1:M, k-1]
            b = b1 + b2
            U[1:M, k] = np.linalg.solve(A, b)


        step1 = []
        for k in range(round(19 / dt), N):
            step1.append(U[round(M/2), k])

        length = (171 - 19) / 0.5
        step_true = perfect_U[:round(length)]
        s = 0
        for i in range(0, round(length)):
            s += (step1[round(i * 0.5 / dt)] - step_true[i])**2

        aHsdict[(ax, Hx)] = s

print(sorted(aHsdict.items(), key=lambda x: x[1], reverse=True))

# plt.plot(tspan, step1)
# plt.plot(tspan, perfect_U[:len(tspan)])
# plt.show()
