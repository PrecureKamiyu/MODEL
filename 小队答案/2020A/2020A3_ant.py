import numpy as np
import random as rd
from scipy import sparse

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

def BinarySearch(T1, T6, T7, T8, vlist, low, high):
    if high > low:
        mid = (high + low) // 2
        [tspan, step] = calculateCurve(T1, T6, T7, T8, vlist[mid])
        # 拟合
        coefficients = np.polyfit(tspan, step, 10)

        # 求斜率
        deltat = 0.01
        maxK = 0
        for i in np.arange(0, l0 / vlist[mid], deltat):
            temp = (np.polyval(coefficients, i + deltat) - np.polyval(coefficients, i)) / deltat
            if temp > maxK:
                maxK = temp
        if maxK > 3 :
            return BinarySearch(T1, T6, T7, T8, vlist, low, mid - 1)
        

        # 求峰值
        tspan_new = np.arange(0, l0 / vlist[mid], 0.01)
        max_temp = max(np.polyval(coefficients, tspan_new))

        if max_temp < 240:
            return BinarySearch(T1, T6, T7, T8, vlist, low, mid - 1)
        if max_temp > 250:
            return BinarySearch(T1, T6, T7, T8, vlist, mid +1, high)
        

        # 求大于217的时间段
        tspan_reverse = np.arange(l0 / vlist[mid], 0, -0.01)
        for t in tspan_new:
            tempL = np.polyval(coefficients, t)
            if tempL > 217:
                tL = t
                break
        for t in tspan_reverse:
            tempR = np.polyval(coefficients, t)
            if tempR > 217:
                tR = t
                break
        if tR - tL < 40:
            return BinarySearch(T1, T6, T7, T8, vlist, low, mid - 1)
        if tR - tL > 90:
            return BinarySearch(T1, T6, T7, T8, vlist, mid +1, high)


        # 求上升阶段150-190所用时长
        for t in tspan_new:
            temp150 = np.polyval(coefficients, t)
            if temp150 > 150:
                t150 = t
                break
        for t in tspan_new:
            temp190 = np.polyval(coefficients, t)
            if temp190 > 190:
                t190 = t
                break
        if t190 - t150 < 60:
            return BinarySearch(T1, T6, T7, T8, vlist, low, mid - 1)
        if t190 - t150 < 120:
            return BinarySearch(T1, T6, T7, T8, vlist, mid +1, high)
        

        # 若在上述过程中没有发生任何跳转
        # 向右半部分进行进一步搜索
        return BinarySearch(T1, T6, T7, T8, vlist, mid, high)
    else:
        print("+++++++++++++++++++++++++")
        return vlist[low]


def getSquare(T1, T6, T7, T8, v):
    # 焊炉总长度
    l0 = 435.5

    [tspan, step] = calculateCurve(T1, T6, T7, T8, v)

    coefficients = np.polyfit(tspan, step, 10)
    # 求峰值
    tspan_new = np.arange(0, l0 / v, 0.01)
    list1 = np.polyval(coefficients, tspan_new)
    max_temp = max(list1)
    max_index = np.inf
    for i in np.arange(l0 / v, 0, -0.01):
        if (abs(np.polyval(coefficients, i) - max_temp)) < 1e-5:
            max_index = i
            break

    # 求大于217的时间段
    for t in tspan_new:
        tempL = np.polyval(coefficients, t)
        if tempL > 217:
            tL = t
            break
    tspan_o = np.arange(tL, max_index, 0.01)
    step_o = np.polyval(coefficients, tspan_o)
    return np.trapz(step_o, tspan_o) - (max_index - tL) * np.polyval(coefficients, tL)



def optimizefunc(x):
    
    vspan = np.arange(65 / 60, 100 / 60, 0.01 / 60)

    v_max = BinarySearch(20 * x[0] + 165, 20 * x[1] + 185, 20 * x[2] + 225, 20 * x[3] + 245,  vspan, 0, len(vspan))

    return getSquare(20 * x[0] + 165, 20 * x[1] + 185, 20 * x[2] + 225, 20 * x[3] + 245, v_max)
    


def bet(T, ant):
    """
    @name: 赌轮盘算法求蚂蚁路径
    @params:
    T: 某一点往下一列的信息素阵列
    ant: 一只走到该点的蚂蚁（包括路径）
    @return: 一只更新完下一路径的蚂蚁
    """
    # 获得标准化的信息素阵列
    total = sum(T)
    TS = [t/total for t in T]

    # 获得累加概率阵列
    TR = []
    for i in range(0, 10):
        TR.append(sum(TS[:i+1]))
    
    # 对于该蚂蚁用赌轮盘算法求出对应路径选择
    rand = rd.random()
    for k in range(0, 10):
        if rand < TR[k]:
            ant_root = k
            break
    ant.append(ant_root)
def decode(Ant, p):
    """
    @name: 解码函数
    @params:
    p: 表示精度
    """
    path2digit = []
    for ant in Ant:
        digit = 0
        for i in range(1, p+1):
            digit += ant[i] * 10 ** (-i)
        path2digit.append( (ant[0], round(digit, p)) )
    return path2digit
def encode(elite, p):
    """
    @name: 编码函数
    @params: 精英蚂蚁解
    @return: 精英编码
    @use: 0.569 -> [5, 6, 9]
    """
    code = []
    for x in elite:
        value = x[1]
        decimal_str = str(value)[2:] # 去掉小数点前面的部分
        decimal_str = decimal_str.ljust(p, '0') # 如果长度不足p位，用0补齐
        result = [int(d) for d in decimal_str] # 将字符串转换为整数列表
        code.append(result)
    return code
def best_around(best, K):
    """
    @name:局部搜索最优路径的边缘路径
    @params:
    best: 最优路径阵列
    K: 全局权值
    @return: 返回一个包含t0,T1的增量矩阵
    """
    num = []
    for i in range(0, len(best)):
        str0 = ""
        for j in range(0, i+1):
            str0 += str(best[j])
        num.append(int(str0))
    nump = np.array(num) + 1
    numn = np.array(num) - 1

    best_t0 = np.zeros(10)
    best_T1 = np.zeros((len(best)-1, 10, 10))

    if nump[0] != 10:
        best_t0[nump[0]] += K/4
    if numn[0] != -1:
        best_t0[numn[0]] += K/4


    for i in range(1, len(nump)):
        if nump[i] != 10 ** (i + 1):
            n = nump[i] % 10
            m = (nump[i] // 10) % 10
            best_T1[(i-1, m, n)] += K/4
        if nump[i] != -1:
            n = numn[i] % 10
            m = (numn[i] // 10) % 10
            best_T1[(i-1, m, n)] += K/4
    T = [best_t0, best_T1]
    return T
def DACA(optfunc, p, num_ant, loop_times, K, Q, J, rho):
    """
    @name: 十进制蚁群算法
    @params: 
    optfunc: 优化函数
    p: 精度要求列表
    @return: 返回最优解
    """
    elitist_num = int(num_ant * 0.2)
    num_x = len(p) # 未知数的个数

    # 初始化信息素矩阵
    # 初始点到十分位的信息素
    t0 = np.ones((num_x, 10))
    # 十分位到百分位 ~ 倒数第二位到末位 的信息素
    T1temp = []
    for pi in p:
        T1temp.append(np.ones((pi - 1, 10, 10)))
    T1 = np.array(T1temp, dtype=object)

    # 开始循环
    for j in range(0, loop_times):

        # 初始化蚂蚁列表
        Ant = []
        for i in range(0, num_ant):
            Ant.append([i])
        
        for x in range(0, num_x):
            # 获得第一轮的十分位蚂蚁阵列
            for ant in Ant:
                bet(t0[x], ant)

            # 循环p-1次更新完蚂蚁阵列
            for col in range(0, p[x]-1):
                
                for ant in Ant:
                    posi = ant[-1]
                    tn = T1[x][col, posi, :]
                    bet(tn, ant)

        # 根据蚂蚁路径阵列进行解码
        ant_x = []
        for i in range(0, num_ant):
            ant_x.append([i])
        for x in range(0, num_x):
            begin = sum(p[:x]) + 1
            end = sum(p[:x+1]) + 1
            temp_code = [ant[0:1] + ant[begin: end] for ant in Ant]
            temp_x = decode(temp_code, p[x])
            for y in range(0, num_ant):
                ant_x[y].append(temp_x[y][1])

        # 根据解码得到的数据对应的函数值进行排序(从小到大)
        sort_x = sorted(ant_x, key= lambda x: optfunc(x[1:num_x+1]))

        # 采取精英策略，对于获得的解进行编码，构造 增量信息素矩阵       
        elite_x = sort_x[ :elitist_num]

        optvalue = [optfunc(x[1:num_x+1]) for x in elite_x]
        best = optvalue[0]

        code = []
        for x in range(0, num_x):
            codei = encode([elite[x: x+2] for elite in elite_x], p[x])
            code.append(codei)

        # code[[[1,2,3],[4,5,6]],
        #      [[1,2,5,9],[8,7,6,2]]]

        best_path = [a[0] for a in code]
        # best_path [[3, 7, 4], [4, 8, 9, 4]]

        # K 权重系数    Q 幂底系数(Q<1)   J 比例系数

        # 获得初始点到十分位的信息素增量
        delta0 = np.zeros((num_x,10))
        for x in range(0, elitist_num):
            for y in range(0, num_x):
                delta0[y][code[y][x][0]] += K * Q ** (J * (optvalue[x] - best))
        
        # 获得十分位到百分位 ~ 倒数第二位到末位 的信息素增量
        delta1_temp = []
        for pi in p:
            delta1_temp.append(np.zeros((pi - 1, 10, 10)))
        delta1 = np.array(delta1_temp, dtype=object)
        for y in range(0, num_x):
            for i in range(0, p[y] - 1):
                for x in range(0, elitist_num):
                    delta1[y][(i, code[y][x][i], code[y][x][i+1])] += K * Q ** (J * (optvalue[x] - best))
        
        # 获得局部搜索边缘权值矩阵     
        around0 = []
        around1_temp = []
        for path in best_path:
            temp_T = best_around(path, K)
            around0.append(temp_T[0])
            around1_temp.append(temp_T[1])
        around1 = np.array(around1_temp, dtype=object)

        # 挥发系数 rho

        for x in t0:
            x *= (1- rho)
        t0 =  t0 + delta0 + around0

        for x in T1:
            x *= (1 - rho)
        T1 =  T1 + delta1 + around1   

    """
    输出最终的信息素矩阵
    """
    print(t0)
    # print(T1)
    for x in range(0,num_x):
        print("x"+str(x) + "------------------")
        epsilon = 5 * K
        print("0->>1:", end='  ')
        for i in range(0, len(t0[x])):
            if(t0[x][i] > epsilon):
                print("0->" + str(i), end=' ')
        print("")
        for i in range(0, len(T1[x])):
            print(str(i+1) +"->>"+  str(i+2) + ":", end='  ')
            for j in range(0, 10):
                for k in range(0, 10):
                    if(T1[x][(i,j,k)] > epsilon):
                        print(str(j) + "->" + str(k), end=' ')
            print("")


DACA(optimizefunc, p=[1, 1, 1, 1], num_ant=20, loop_times=5, K=5, Q=0.8, J=100, rho=0.4)