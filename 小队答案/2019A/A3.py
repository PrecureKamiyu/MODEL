import numpy as np
import csv
import pylab as plt

# 得出关于rho的P函数
P = []
E_inverse = []

with open("data1.csv", "r") as csvfile:
    reader = csv.reader(csvfile)  # 创建一个reader对象
    for row in reader:
        P.append(float(row[0]))
        E_inverse.append(1/float(row[1]))
rho_list = []
for i in range(0, 200):
    rho_list.append(0.850 * np.exp(-np.trapz(E_inverse[i: 201], P[i: 201])))
rho_list.append(0.850)
for i in range(202, 402):
    rho_list.append(0.850 * np.exp(np.trapz(E_inverse[200: i], P[200: i])))

rho_new = np.arange(0.8043, 0.8825, 0.0001)
coefficients = np.polyfit(rho_list, P, 10)


"""
总参数表
"""
rho0 = 0.8711223  # 初始高压油管内压强
L = 500
D = 10
d = 1.4
V = np.pi * (D/2)**2 * L    # 高压油管体积
P1 = 160
C = 0.85
A = np.pi * (d/2)**2     # A口对应的面积

"""
进油口A部分
"""
# A参数


P_new = np.polyval(coefficients, rho_new)


wheelTheta = []
wheelRho = []

with open("data2.csv", "r") as csvfile:
    reader = csv.reader(csvfile)  # 创建一个reader对象
    for row in reader:
        wheelTheta.append(float(row[0]))
        wheelRho.append(float(row[1]))

wheelThetaNew = np.arange(0, 6.271, 0.0001)

coefficient = np.polyfit(wheelTheta, wheelRho, 10)

wheelRhoNew = np.polyval(coefficient, wheelThetaNew)


def getRho(theta):
    return np.polyval(coefficient, theta)

# plt.rc('text', usetex=True)
# plt.rc('font', size=15)
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$\rho$', rotation=0)
# plt.plot(wheelTheta, wheelRho, 'o')  # 显示原始点数据
# plt.plot(wheelThetaNew, wheelRhoNew, label="Curve fit result")  # 显示拟合数据
# plt.legend()
# plt.show()

# #作出曲线轮廓
# wheelX = []
# wheelY = []
# for i in range(0, 62701, 1):
#     print(i)
#     wheelX.append(wheelRhoNew[i] * np.cos(wheelThetaNew[i]))
#     wheelY.append(wheelRhoNew[i] * np.sin(wheelThetaNew[i]))
# plt.rc('text', usetex=True)
# plt.rc('font', size=15)
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$', rotation=0)
# plt.plot(wheelX, wheelY)
# plt.show()

# 找初始角度
# rhoStart = 5.13852
# close = np.inf
# closeIndex = 0
# for i in range(0, 62701, 1):
#     if (abs(rhoStart - wheelRhoNew[i])) < close:
#         close = abs(rhoStart - wheelRhoNew[i])
#         closeIndex = i
# print(wheelRhoNew[closeIndex])
# print(wheelThetaNew[closeIndex])
# 求得wheelThetaNew[closeIndex]为1.4409
# 在此初始角度开始逆时针旋转


def Q_in(pL, pR, rhoL):
    if pL > pR:
        return C * A * np.sqrt(2 * (pL - pR) / rhoL)
    else:
        return 0


"""
喷油嘴B部分
"""
# B参数
theta0 = np.pi / 20
d2 = 1.4
d3 = 2.5
Pen = 0.1013

updist = []
downdist = []
with open("data3.csv", "r") as csvfile:
    reader1 = csv.reader(csvfile)  # 创建一个reader对象
    for row in reader1:
        updist.append(float(row[1]))
        downdist.append(float(row[4]))
uptime = np.linspace(0, 0.44, 45)
downtime = np.linspace(2.01, 2.45, 45)
upcoe = np.polyfit(uptime, updist, 10)
downcoe = np.polyfit(downtime, downdist, 10)


def Q_out(P, t, rho):
    t_temp = t % 100

    # 针阀闭合
    if t_temp >= 2.45:
        return 0

    # 针阀的三种情况
    elif t_temp <= 0.44:
        h = np.polyval(upcoe, t_temp)
    elif t_temp <= 2.00:
        h = 2
    else:
        h = np.polyval(downcoe, t_temp)

    # 计算圆台侧面积
    S_Ring = np.pi * (h * np.sin(theta0) * np.cos(theta0) +
                      d3) * h * np.sin(theta0)
    S_bottom = np.pi * (d2 / 2)**2

    # 取较小的面积作为有效面积
    A = min(S_bottom, S_Ring)

    return C * A * np.sqrt(2 * (P - Pen) / rho)

def Q_outD(pR, rhoR):
    if pR >= 100:
        return C * A * np.sqrt(2 * (pR - 0.5)/ rhoR)
    else:
        return 0


"""
主函数部分
"""
# 偏差值总和列表
omegas_dict = {}

for omega in np.arange(0.435, 0.445, 0.0005):

    print(omega)

    # 重置参数部分
    dt = 0.01
    t = 0
    rhoR = 0.85       # 100Pa
    rhoL = 0.85
    flag = 1
    thetaInit = 2.63
    theta = thetaInit
    pL = 100
    pR = 100
    mInit = 92.32863
    m = mInit
    v = 108.62191
    # 存放每个时间点对应的P值
    tP_dict = {}

    # 开始差分
    while t <= 1000:
        theta -= omega * dt
        theta %= 2 * np.pi
        inTemp = Q_in(pL, pR, rhoL)
        outTemp = 2 * Q_out(pR, t, rhoR) + Q_outD(pR, rhoR)
        rhoPrime = (inTemp * rhoL - outTemp * rhoR) / V
        rhoR += rhoPrime * dt
        
        # 判断是否刷新油量
        if theta < np.pi and flag == 0:
            m = mInit
            flag = 1
        if theta > np.pi:
            flag = 0
        
        if pL > pR:
            m -= rhoL * inTemp * dt

        v -= np.pi * (5/2)**2 * (getRho(theta) - getRho((theta + omega * dt) % (2*np.pi)))
        rhoL = m / v
        pL = np.polyval(coefficients, rhoL)
        pR = np.polyval(coefficients, rhoR)

        tP_dict[t] = pR
        t += dt

    # 重置偏差值总和为0
    s = 0
    # 计算、存储偏差值
    for t, P in tP_dict.items():
        s += (P - 100)**2
    omegas_dict[omega] = s


# show()部分
Key = []
Value = []
for k, v in omegas_dict.items():
    Key.append(k)
    Value.append(v)
plt.rc('text', usetex=True)
plt.rc('font', size=15)
plt.xlabel('$ T $')
plt.ylabel('$ s $', rotation=0)
plt.plot(Key, Value, 'o')  # 显示原始点数据
plt.show()