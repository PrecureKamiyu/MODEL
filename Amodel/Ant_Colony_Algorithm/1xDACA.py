import numpy as np
import random as rd

def optimizefunc(x):
    # return (x - 0.396086) ** 2 / 100000
    return abs(abs(abs(x - 0.5) - 0.25)-0.125)


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


def DACA(optfunc):
    """
    @name: 十进制蚁群算法
    @params: 优化函数
    @return: 返回最优解
    """
    num_ant = 1000
    elitist_num = 50
    loop_times = 20
    p = 6    # precision

    # 初始化信息素矩阵
    # 初始点到十分位的信息素
    t0 = np.ones(10)

    # 十分位到百分位 ~ 倒数第二位到末位 的信息素
    T1 = np.ones((p-1, 10, 10))

    # 开始循环
    for j in range(0, loop_times):

        # 初始化蚂蚁列表
        Ant = []
        for i in range(0, num_ant):
            Ant.append([i])
        
        # 获得第一轮的十分位蚂蚁阵列
        for ant in Ant:
            bet(t0, ant)

        # 循环p-1次更新完蚂蚁阵列
        for col in range(0, p-1):
            
            for ant in Ant:
                posi = ant[-1]
                tn = T1[col, posi, :]
                bet(tn, ant)

        # 根据蚂蚁路径阵列进行解码
        ant_x = decode(Ant, p)

        # 根据解码得到的数据对应的函数值进行排序(从小到大)
        sort_x = sorted(ant_x, key= lambda x: optfunc(x[1]))
        
        # 采取精英策略，对于获得的解进行编码，构造 增量信息素矩阵
        elite_x = sort_x[ :elitist_num]

        optvalue = [optfunc(x[1]) for x in elite_x]
        best = optvalue[0]

        code = encode(elite_x, p)
        best_path = code[0]

        # K 权重系数    Q 幂底系数(Q<1)   J 比例系数
        K = 5;     Q = 0.8;    J = 1000

        # 获得初始点到十分位的信息素增量
        delta0 = np.zeros(10)
        for x in range(0, elitist_num):
            delta0[code[x][0]] += K * Q ** (J * (optvalue[x] - best))
        
        # 获得十分位到百分位 ~ 倒数第二位到末位 的信息素增量
        delta1 = np.zeros((p-1, 10, 10))
        for i in range(0, p-1):
            for x in range(0, elitist_num):
                delta1[(i, code[x][i], code[x][i+1])] += K * Q ** (J * (optvalue[x] - best))

        # 获得局部搜索边缘权值矩阵
        T = best_around(best_path, K)

        rho = 0.4 # 挥发系数
        t0 = (1 - rho) * t0 + delta0 + T[0]
        T1 = (1 - rho) * T1 + delta1 + T[1]


    """
    输出最终的信息素矩阵
    """
    # print(t0)
    # print(T1)


    # 输出大于epsilon的信息素转移项，构成最终转移路径矩阵
    epsilon = 10
    print("0->>1:", end='  ')
    for i in range(0, len(t0)):
        if(t0[i] > epsilon):
            print("0->" + str(i), end=' ')
    print("")
    for i in range(0, len(T1)):
        print(str(i+1) +"->>"+  str(i+2) + ":", end='  ')
        for j in range(0, 10):
            for k in range(0, 10):
                if(T1[(i,j,k)] > epsilon):
                    print(str(j) + "->" + str(k), end=' ')
        print("")



# 运行
DACA(optimizefunc)

# 当函数在0.25,0.75处取得最小值时，转移矩阵最大项为：
# 01: 0->2  0->7      12:  2->5   7->5     23:  5->0



