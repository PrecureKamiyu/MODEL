import numpy as np
import random as rd


def bet(T, n):
    """
    @name: 赌轮盘算法求蚂蚁路径列表
    @params:
    T: 某一点往下一列的信息素阵列
    n: 蚂蚁数量
    @return: 每只蚂蚁最终选择的路径
    """
    # 获得标准化的信息素阵列
    total = sum(T)
    TS = [t/total for t in T]

    # 获得累加概率阵列
    TR = []
    for i in range(0, 10):
        TR.append(sum(TS[:i+1]))
    
    # 对于每一只蚂蚁用赌轮盘算法求出对应路径选择
    ant = []
    for i in range(0, n):
        rand = rd.random()
        for k in range(0, 10):
            if rand < TR[k]:
                ant.append(k)
                break
    return ant

