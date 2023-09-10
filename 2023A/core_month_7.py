import numpy as np
import pandas as pd
import random
import openpyxl as op
"""
STEP1
"""

data = pd.read_excel('附件.xlsx', sheet_name= 'Sheet1')

# 中心坐标
x0 = []
y0 = []
for row in data.iloc:
    x0.append(row[0])
    y0.append(row[1])


h_collect = 8
d = 4
D_O = 7
mid = 80
h = mid + h_collect / 2
xR = 0
yR = 0
a = 6
b = 6
eta_ref = 0.92
# 建立一个面板字典，key为面板编号，存储  从中心到反射点的向量  信息

mirror = {}
for i in range(0, len(x0)):
    n = [x0[i] - xR, y0[i] - yR, d - mid]
    n = n / np.linalg.norm(n)
    mirror[i] = n

# 根据半径分类第几排
mirror_R = {}
for i in range(0, len(x0)):
    mirror_R[i] = round((x0[i]**2 + y0[i]**2)**0.5)

# 获得special间断点列表
special = [0]
front = 108
for i in range(0, len(x0)):
    if mirror_R[i] == front:
        continue
    else:
        front = mirror_R[i]
        special.append(i)
special.append(len(x0))

# 判断并返回前排列表函数
def check_and_return_front(check):
    for i in range(1, len(special)):
        if check < special[i] and i == 1:
            return []
        elif check < special[i]:
            return range(special[i - 2], special[i - 1])


Day_list = [121]
ST_list = [9, 10.5, 12, 13.5, 15]

# 以下写某两组个循环
# 先以1.21的9点为例

twelve_data = []
for Day in Day_list:
    five_eta = []
    five_cos = []
    five_sb = []
    five_trunc = []
    five_EdivideS = []
    for STime in ST_list:
        D = Day
        ST = STime

        phi = np.deg2rad(39.4)
        omega = np.pi / 12 * (ST - 12)
        sindelta = np.sin(2 * np.pi * D / 365) * np.sin(2 * np.pi * 23.45 / 360)

        cosdelta = (1 - sindelta**2)**0.5
        # alphas
        sinalphas = cosdelta * np.cos(phi) * np.cos(omega) + sindelta * np.sin(phi)
        cosalphas = (1 - sinalphas**2)**0.5
        # gammas
        cosgammas = (sindelta - sinalphas * np.sin(phi))/(cosalphas * np.cos(phi))
        singammas = np.sign(ST - 12) * (max(0, 1 - cosgammas**2))**0.5

        # 确定所有面板的法向量

        xn = - cosalphas * singammas
        yn = - cosalphas * cosgammas
        zn = - sinalphas

        # 入射光线向量
        lambdai = [xn, yn, zn]

        # 法向量字典
        n_dict = {}
        for i in range(0, len(x0)):
            n_dict[i] = - (mirror[i] + lambdai)
            n_dict[i] /=  np.linalg.norm(n_dict[i])

        """
        STEP2
        """
        # 开始在all面板中monte carlo


        per_mirror_MC_times = 200
        eta_list = []
        eta_cos_list = []
        eta_sb_list = []
        eta_trunc_list = []


        # 面板进行循环计算其各种效率
        for i in range(0, len(x0)):
            print(i)
            blocked_times = 0
            eta_dot_trunc_list = []
            for j in range(0, per_mirror_MC_times):
                # 生成随机点
                xA = random.uniform(-a/2, a/2)
                yA = random.uniform(-b/2, b/2)
                nA = n_dict[i]

                # 求最近前板编号B
                front_list = check_and_return_front(i)
                dAB = np.inf
                B = -1
                for k in front_list:
                    dAB_temp = ((x0[i] - x0[k])**2 + (y0[i] - y0[k])**2)**0.5
                    if dAB_temp <= dAB:
                        dAB = dAB_temp
                        B = k
                
                # 转换A点的坐标
                mid_theta = np.arctan(- nA[0] / nA[1])
                if (mid_theta == 0):
                    mid_sinphi = - nA[1]
                    mid_cosphi = nA[2]
                else:
                    mid_sinphi = nA[0] / np.sin(mid_theta)
                    mid_cosphi = nA[2]
                
                x = xA * mid_cosphi - yA * mid_cosphi * np.sin(mid_theta) + x0[i]
                y = xA * np.sin(mid_theta) + yA * mid_cosphi * np.cos(mid_theta) + y0[i]
                z = yA * mid_sinphi + d


                # 先判断入射光线是否经过圆柱塔
                # 比例系数k   入射光线与平面 z = h 的交点 Nh
                k = (h - z) / lambdai[2]
                x_h = x + lambdai[0] * k
                y_h = y + lambdai[1] * k

                # 先计算入射光线与中心线AO在xOy维度内的夹角
                # 最小余弦值计算
                min = ((x**2 + y**2) - (D_O/2)**2)**0.5 / (x**2 + y**2)**0.5

                cos_theta = (x * lambdai[0] + y * lambdai[1]) / ((x**2 + y**2)**0.5 * (lambdai[0]**2 + lambdai[1]**2)**0.5)

                # flag 为是否计算trunc和其他效率的标志变量
                flag = 1
                if cos_theta < min:
                    flag = 1
                else:
                    if x**2 + y**2 >= (D_O/2)**2 and (x - x_h)**2 + (y - y_h)**2 <= (x**2 + y**2) - (D_O/2)**2:
                        flag = 1
                    else:
                        blocked_times += 1
                        flag = 0

                # 根据A点坐标，确定反射主光线
                k = -2 * (nA[0] * lambdai[0] + nA[1] * lambdai[1] + nA[2] * lambdai[2])
                xo = lambdai[0] + k * nA[0]
                yo = lambdai[1] + k * nA[1]
                zo = lambdai[2] + k * nA[2]
                lambdao = (xo, yo, zo)
                
                # 接下来判断是否被B板挡住（反射光）
                # B板中心点坐标(xB, yB, d)
                if B != -1:
                    xB = x0[B]
                    yB = y0[B]
                    nB = n_dict[B]
                    m = ((xB - x) * nB[0] + (yB - y) * nB[1] + (d - z) * nB[2]) / (lambdao[0] * nB[0] + lambdao[1] * nB[1] + lambdao[2] * nB[2])

                    # 求得打在B板上反射光线的坐标(xb_earth, yb_earth, zb_earth)  平移了d
                    xb_earth = x + m * lambdao[0] - xB
                    yb_earth = y + m * lambdao[1] - yB
                    zb_earth = z + m * lambdao[2] - d
                    
                    # 求B板对应的theta和phi
                    mid_theta = np.arctan(- nB[0] / nB[1])
                    if (mid_theta == 0):
                        mid_sinphi = - nB[1]
                        mid_cosphi = nB[2]
                    else:
                        mid_sinphi = nB[0] / np.sin(mid_theta)
                        mid_cosphi = nB[2]

                    x_Bboard = np.cos(mid_theta) * xb_earth + np.sin(mid_theta) * yb_earth
                    y_Bboard = - mid_cosphi * np.sin(mid_theta) * xb_earth + mid_cosphi * np.cos(mid_theta) * yb_earth + mid_sinphi * zb_earth
                    z_Bboard = mid_sinphi * np.sin(mid_theta) * xb_earth - mid_sinphi * np.cos(mid_theta) * yb_earth + mid_cosphi * zb_earth

                    if x_Bboard >= -3 and x_Bboard <= 3 and y_Bboard >= -3 and y_Bboard <= 3:
                        flag = 0
                        blocked_times += 1

                """
                判断入射光是否被挡住
                """
                # 首先找出临近的点(且这些点的位置比i板位置还南)
                around_list = []
                for r in range(0, len(x0)):
                    if (x0[i]-x0[r]) **2 + (y0[i] - y0[r])**2 <= 20**2 and y0[i] > y0[r]: 
                        around_list.append(r)
                for r in around_list:
                    m = ((x0[i] - x0[r]) * nA[0] + (y0[i] - y0[r]) * nA[1]) / (lambdai[0] * nA[0] + lambdai[1] * nA[1] + lambdai[2] * nA[2])
                    xb_ear = x0[r] + m * lambdai[0] - x0[i]
                    yb_ear = y0[r] + m * lambdai[1] - y0[i]
                    zb_ear = d + m * lambdai[2] - d

                    mid_theta = np.arctan(- nA[0] / nA[1])
                    if (mid_theta == 0):
                        mid_sinphi = - nA[1]
                        mid_cosphi = nA[2]
                    else:
                        mid_sinphi = nA[0] / np.sin(mid_theta)
                        mid_cosphi = nA[2]

                    # 计算前板中心在后板上的投影点（在后板坐标系中）
                    x_Aboard = np.cos(mid_theta) * xb_ear + np.sin(mid_theta) * yb_ear
                    y_Aboard = - mid_cosphi * np.sin(mid_theta) * xb_ear + mid_cosphi * np.cos(mid_theta) * yb_ear + mid_sinphi * zb_ear
                    z_Aboard = mid_sinphi * np.sin(mid_theta) * xb_ear - mid_sinphi * np.cos(mid_theta) * yb_ear + mid_cosphi * zb_ear

                    if xA <= x_Aboard + a/2 and xA >= x_Aboard - a/2 and yA <= y_Aboard + b/2 and yA >= y_Aboard - b/2:
                        # 蒙特卡罗点在阴影范围中
                        flag = 0
                        blocked_times += 1

                # 接下来计算trunc

                if flag == 1:
                    # 利用光锥,算该点的截断效率

                    light_MC_times = 100
                    in_times = 0
                    out_times = 0
                    for m in range(0, light_MC_times):
                        # 蒙特卡罗生成随机角度
                        tau = random.uniform(0, 2 * np.pi)
                        sigma = random.uniform(0, 0.465 * 0.001)

                        # 在光锥坐标系中随机散光的向量
                        So = [np.sin(sigma) * np.cos(tau), np.sin(sigma) * np.sin(tau), np.cos(sigma)]

                        # 由lambdao和So确定So在主坐标系中的表示S
                        if (lambdao[1] == 0):
                            mid_m = 0
                            mid_n = 1
                        else:
                            mid_m = abs(lambdao[1]) / (lambdao[0] ** 2 + lambdao[1] ** 2)**0.5
                            mid_n = (- lambdao[0] * abs(lambdao[1]) / lambdao[1]) / (lambdao[0]**2 + lambdao[1] ** 2)**0.5

                        S = [So[0] * mid_m - So[1] * mid_n * lambdao[2] + So[2] * lambdao[0],
                            So[0] * mid_n - So[1] * mid_m * lambdao[2] + So[2] * lambdao[1],
                            So[1] * mid_m * lambdao[1] - So[1] * mid_n * lambdao[0] + So[2] * lambdao[2]]

                        # 比例系数ko
                        ko84 = (h - z) / S[2]
                        x_h84 = x + ko84 * S[0]
                        y_h84 = y + ko84 * S[1]

                        ko76 = (h - h_collect - z) / S[2]
                        x_h76 = x + ko76 * S[0]
                        y_h76 = y + ko76 * S[1]

                        # 先计算So与中心线AO在xOy维度内的夹角
                        # 最小余弦值计算
                        min = ((x**2 + y**2) - (D_O/2)**2)**0.5 / (x**2 + y**2)**0.5
                        cos_theta = - (x * S[0] + y * S[1]) / ((x**2 + y**2)**0.5 * (S[0]**2 + S[1]**2)**0.5)

                        if cos_theta < min:
                            out_times += 1
                        else:
                            if x**2 + y**2 >= (D_O/2)**2 and (x - x_h84)**2 + (y - y_h84)**2 <= (x**2 + y**2) - (D_O/2)**2:
                                out_times += 1
                            else:
                                if x**2 + y**2 >= (D_O/2)**2 and (x - x_h76)**2 + (y - y_h76)**2 <= (x**2 + y**2) - (D_O/2)**2:
                                    in_times += 1
                                else:
                                    out_times += 1

                    trunc = in_times / light_MC_times
                    eta_dot_trunc_list.append(trunc)

            eta_sb = (per_mirror_MC_times - blocked_times) / per_mirror_MC_times
            
            if len(eta_dot_trunc_list) == 0:
                eta_trunc = 0
            else:
                eta_trunc = sum(eta_dot_trunc_list) / len(eta_dot_trunc_list)
            
            # 通过法向量和入射光线向量确定eta_cos
            cos_theta = - (lambdai[0] * n_dict[i][0] + lambdai[1] * n_dict[i][1] + lambdai[2] * n_dict[i][2]) / (np.linalg.norm(lambdai) * np.linalg.norm(n_dict[i]))
            eta_cos = cos_theta


            d_HR = ((x0[i] - xR)**2 + (y0[i] - yR)**2 + (d - mid)**2)**0.5
            eta_at = 0.99321 - 0.0001176 * d_HR + 1.97 * (1e-8) * d_HR ** 2

            eta = eta_sb * eta_trunc * eta_cos * eta_at * eta_ref


            eta_list.append(eta)
            eta_cos_list.append(eta_cos)
            eta_sb_list.append(eta_sb)
            eta_trunc_list.append(eta_trunc)
            print(eta, eta_cos,eta_sb, eta_trunc)

        avg_eta = sum(eta_list) / len(eta_list)
        avg_eta_cos = sum(eta_cos_list)  / len(eta_cos_list)
        avg_eta_sb = sum(eta_sb_list) / len(eta_sb_list)
        avg_eta_trunc = sum(eta_trunc_list) / len(eta_trunc_list)


        G0 = 1.366
        H = 3
        a = 0.4237 - 0.00821 * (6 - H)**2
        b = 0.5055 + 0.00595 * (6.5 - H)**2
        c = 0.2711 + 0.01858 * (2.5 - H)**2

        DNI = G0 * (a + b * np.exp(-c / sinalphas))

        # 这里因为所有定日镜面积一致   简略写为  a * b
        E_field = DNI * a * b * sum(eta_list)
        sum_S = a * b * len(x0)

        five_eta.append(avg_eta)
        five_cos.append(avg_eta_cos)
        five_sb.append(avg_eta_sb)
        five_trunc.append(avg_eta_trunc)
        five_EdivideS.append(E_field/sum_S)
    
    twelve_data.append([sum(five_eta)/5,
                        sum(five_cos)/5,
                        sum(five_sb)/5,
                        sum(five_trunc)/5,
                        sum(five_EdivideS)/5])

workbook = op.Workbook()
worksheet1 = workbook.active
worksheet1.title = "Sheet1"
data = [["平均光学效率", "平均余弦效率", "平均阴影遮挡效率", "平均截断效率", "单位面积镜面平均输出热功率"]]

for i in twelve_data:
    data.append(i)

for row in data:
    worksheet1.append(row)

workbook.save(filename="result7.xlsx")
