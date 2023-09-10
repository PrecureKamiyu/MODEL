import numpy as np
import random
import openpyxl as op

"""
STEP1
"""
yR = -197.03
a = 6.365
b0 = 4.386
d0 = 4.341
delta_r = 12.96
delta_d = 0.001214
delta_b = -0.001818


def return_dot_list(yR, delta_r, a):
    r = 108
    count = 0
    dot_list = []
    while r <= 350 + yR:
        N = int(np.floor((2 * np.pi * r) / (a + 5)))
        delta_theta = 2 * np.pi / N
        off = delta_theta * random.uniform(0, 1)
        for i in range(0, N):
            dot_list.append((r * np.cos(delta_theta * i + off),
                            r * np.sin(delta_theta * i + off) + yR, count))
        r += delta_r
        count += 1
    while r <= 350 - yR:
        N = int(np.floor((2 * np.pi * r) / (a + 5)))
        delta_theta = 2 * np.pi / N
        off = delta_theta * random.uniform(0, 1)
        for i in range(0, N):
            x = r * np.cos(delta_theta * i + off)
            y = r * np.sin(delta_theta * i + off) + yR
            if x**2 + y ** 2 <= 350**2:
                dot_list.append((x, y, count))
        r += delta_r
        count += 1
    return dot_list


list = return_dot_list(yR, delta_r, a)
x0 = []
y0 = []
c0 = []
N = len(x0)
for i in list:
    x0.append(i[0])
    y0.append(i[1])
    c0.append(i[2])
h_collect = 8
D_O = 7
mid = 80
h = mid + h_collect / 2
xR = 0
eta_ref = 0.92
mirror = {}
for i in range(0, len(x0)):
    d = d0 + delta_d * c0[i]
    n = [x0[i] - xR, y0[i] - yR, d - mid]
    n = n / np.linalg.norm(n)
    mirror[i] = n
mirror_R = {}
for i in range(0, len(x0)):
    mirror_R[i] = round((x0[i]**2 + y0[i]**2)**0.5)
special = [0]
front = 108
for i in range(0, len(x0)):
    if mirror_R[i] == front:
        continue
    else:
        front = mirror_R[i]
        special.append(i)
special.append(len(x0))


def check_and_return_front(check):
    for i in range(1, len(special)):
        if check < special[i] and i == 1:
            return []
        elif check < special[i]:
            return range(special[i - 2], special[i - 1])


Day_list = [305, 336, 0, 30, 60, 91, 121, 152, 183, 213, 244, 274]
ST_list = [9, 10.5, 12, 13.5, 15]

twelve_data = []
for D in Day_list:
    five_eta = []
    five_cos = []
    five_sb = []
    five_trunc = []
    five_EdivideS = []
    for ST in [9]:
        phi = np.deg2rad(39.4)
        omega = np.pi / 12 * (ST - 12)
        sindelta = np.sin(2 * np.pi * D / 365) * \
            np.sin(2 * np.pi * 23.45 / 360)
        cosdelta = (1 - sindelta**2)**0.5
        # alphas
        sinalphas = cosdelta * np.cos(phi) * \
            np.cos(omega) + sindelta * np.sin(phi)
        cosalphas = (1 - sinalphas**2)**0.5
        # gammas
        cosgammas = (sindelta - sinalphas * np.sin(phi)) / \
            (cosalphas * np.cos(phi))
        singammas = np.sign(ST - 12) * (max(0, 1 - cosgammas**2))**0.5
        xn = - cosalphas * singammas
        yn = - cosalphas * cosgammas
        zn = - sinalphas
        lambdai = [xn, yn, zn]
        n_dict = {}
        for i in range(0, len(x0)):
            n_dict[i] = - (mirror[i] + lambdai)
            n_dict[i] /= np.linalg.norm(n_dict[i])
        """
        STEP2
        """
        per_mirror_MC_times = 200
        eta_list = []
        eta_cos_list = []
        eta_sb_list = []
        eta_trunc_list = []
        for i in range(0, len(x0)):
            print(i)
            blocked_times = 0
            eta_dot_trunc_list = []
            for j in range(0, per_mirror_MC_times):
                b = b0 + c0[i] * delta_b
                xA = random.uniform(-a/2, a/2)
                yA = random.uniform(-b/2, b/2)
                nA = n_dict[i]
                d = d0 + delta_d * c0[i]
                front_list = check_and_return_front(i)
                dAB = np.inf
                B = -1
                for k in front_list:
                    dAB_temp = ((x0[i] - x0[k])**2 + (y0[i] - y0[k])**2)**0.5
                    if dAB_temp <= dAB:
                        dAB = dAB_temp
                        B = k
                mid_theta = np.arctan(- nA[0] / nA[1])
                if (mid_theta == 0):
                    mid_sinphi = - nA[1]
                    mid_cosphi = nA[2]
                else:
                    mid_sinphi = nA[0] / np.sin(mid_theta)
                    mid_cosphi = nA[2]
                x = xA * mid_cosphi - yA * mid_cosphi * \
                    np.sin(mid_theta) + x0[i]
                y = xA * np.sin(mid_theta) + yA * mid_cosphi * \
                    np.cos(mid_theta) + y0[i]
                z = yA * mid_sinphi + d

                k = (h - z) / lambdai[2]
                x_h = x + lambdai[0] * k
                y_h = y + lambdai[1] * k
                min = ((x**2 + (y - yR)**2) - (D_O/2) **
                       2)**0.5 / (x**2 + (y-yR)**2)**0.5
                cos_theta = (x * lambdai[0] + (y-yR) * lambdai[1]) / (
                    (x**2 + (y-yR)**2)**0.5 * (lambdai[0]**2 + lambdai[1]**2)**0.5)
                flag = 1
                if cos_theta < min:
                    flag = 1
                else:
                    if x_h**2 + (y_h-yR)**2 >= (D_O/2)**2 and (x - x_h)**2 + (y - y_h)**2 <= (x**2 + (y-yR)**2) - (D_O/2)**2:
                        flag = 1
                    else:
                        blocked_times += 1
                        flag = 0
                if flag == 1:
                    k = -2 * (nA[0] * lambdai[0] + nA[1] *
                              lambdai[1] + nA[2] * lambdai[2])
                    xo = lambdai[0] + k * nA[0]
                    yo = lambdai[1] + k * nA[1]
                    zo = lambdai[2] + k * nA[2]
                    lambdao = (xo, yo, zo)
                    if B != -1:
                        xB = x0[B]
                        yB = y0[B]
                        nB = n_dict[B]
                        m = ((xB - x) * nB[0] + (yB - y) * nB[1] + (d - z) * nB[2]) / (
                            lambdao[0] * nB[0] + lambdao[1] * nB[1] + lambdao[2] * nB[2])
                        xb_earth = x + m * lambdao[0] - xB
                        yb_earth = y + m * lambdao[1] - yB
                        zb_earth = z + m * lambdao[2] - d
                        mid_theta = np.arctan(- nB[0] / nB[1])
                        if (mid_theta == 0):
                            mid_sinphi = - nB[1]
                            mid_cosphi = nB[2]
                        else:
                            mid_sinphi = nB[0] / np.sin(mid_theta)
                            mid_cosphi = nB[2]
                        x_Bboard = np.cos(mid_theta) * xb_earth + \
                            np.sin(mid_theta) * yb_earth
                        y_Bboard = - mid_cosphi * np.sin(mid_theta) * xb_earth + mid_cosphi * np.cos(
                            mid_theta) * yb_earth + mid_sinphi * zb_earth
                        z_Bboard = mid_sinphi * np.sin(mid_theta) * xb_earth - mid_sinphi * np.cos(
                            mid_theta) * yb_earth + mid_cosphi * zb_earth
                        if x_Bboard >= -3 and x_Bboard <= 3 and y_Bboard >= -3 and y_Bboard <= 3:
                            flag = 0
                            blocked_times += 1
                if flag == 1:
                    around_list = []
                    for r in range(0, len(x0)):
                        if (x0[i]-x0[r]) ** 2 + (y0[i] - y0[r])**2 <= 20**2 and y0[i] > y0[r]:
                            around_list.append(r)
                    for r in around_list:
                        m = ((x0[i] - x0[r]) * nA[0] + (y0[i] - y0[r]) * nA[1]) / \
                            (lambdai[0] * nA[0] + lambdai[1]
                             * nA[1] + lambdai[2] * nA[2])
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
                        x_Aboard = np.cos(mid_theta) * xb_ear + \
                            np.sin(mid_theta) * yb_ear
                        y_Aboard = - mid_cosphi * \
                            np.sin(mid_theta) * xb_ear + mid_cosphi * \
                            np.cos(mid_theta) * yb_ear + mid_sinphi * zb_ear
                        z_Aboard = mid_sinphi * \
                            np.sin(mid_theta) * xb_ear - mid_sinphi * \
                            np.cos(mid_theta) * yb_ear + mid_cosphi * zb_ear
                        if xA <= x_Aboard + a/2 and xA >= x_Aboard - a/2 and yA <= y_Aboard + (b0 + c0[r] * delta_b)/2 and yA >= y_Aboard - (b0 + c0[r] * delta_b)/2:
                            flag = 0
                            blocked_times += 1
                            break
                if flag == 1:
                    light_MC_times = 100
                    in_times = 0
                    out_times = 0
                    for m in range(0, light_MC_times):
                        tau = random.uniform(0, 2 * np.pi)
                        sigma = random.uniform(0, 0.465 * 0.001)
                        So = [np.sin(sigma) * np.cos(tau),
                              np.sin(sigma) * np.sin(tau), np.cos(sigma)]
                        if (lambdao[1] == 0):
                            mid_m = 0
                            mid_n = 1
                        else:
                            mid_m = abs(
                                lambdao[1]) / (lambdao[0] ** 2 + lambdao[1] ** 2)**0.5
                            mid_n = (- lambdao[0] * abs(lambdao[1]) / lambdao[1]
                                     ) / (lambdao[0]**2 + lambdao[1] ** 2)**0.5
                        S = [So[0] * mid_m - So[1] * mid_n * lambdao[2] + So[2] * lambdao[0],
                             So[0] * mid_n - So[1] * mid_m *
                             lambdao[2] + So[2] * lambdao[1],
                             So[1] * mid_m * lambdao[1] - So[1] * mid_n * lambdao[0] + So[2] * lambdao[2]]
                        ko84 = (h - z) / S[2]
                        x_h84 = x + ko84 * S[0]
                        y_h84 = y + ko84 * S[1]
                        ko76 = (h - h_collect - z) / S[2]
                        x_h76 = x + ko76 * S[0]
                        y_h76 = y + ko76 * S[1]
                        min = ((x**2 + (y-yR)**2) - (D_O/2)**2)**0.5 / \
                            (x**2 + (y-yR)**2)**0.5
                        cos_theta = - \
                            (x * S[0] + (y-yR) * S[1]) / ((x**2 + (y-yR)
                                                           ** 2)**0.5 * (S[0]**2 + S[1]**2)**0.5)
                        if cos_theta < min:
                            out_times += 1
                        else:
                            if x_h84**2 + (y_h84-yR)**2 >= (D_O/2)**2 and (x - x_h84)**2 + (y - y_h84)**2 <= (x**2 + (y-yR)**2) - (D_O/2)**2:
                                out_times += 1
                            else:
                                if x_h76**2 + (y_h76-yR)**2 >= (D_O/2)**2 and (x - x_h76)**2 + (y - y_h76)**2 <= (x**2 + (y-yR)**2) - (D_O/2)**2:
                                    in_times += 1
                                else:
                                    out_times += 1
                    trunc = in_times / light_MC_times
                    eta_dot_trunc_list.append(trunc)
            eta_sb = (per_mirror_MC_times - blocked_times) / \
                per_mirror_MC_times
            if len(eta_dot_trunc_list) == 0:
                eta_trunc = 0
            else:
                eta_trunc = sum(eta_dot_trunc_list) / len(eta_dot_trunc_list)
            cos_theta = - (lambdai[0] * n_dict[i][0] + lambdai[1] * n_dict[i][1] + lambdai[2]
                           * n_dict[i][2]) / (np.linalg.norm(lambdai) * np.linalg.norm(n_dict[i]))
            eta_cos = cos_theta

            d_HR = ((x0[i] - xR)**2 + (y0[i] - yR)**2 +
                    (d0 + delta_d * c0[i] - mid)**2)**0.5
            eta_at = 0.99321 - 0.0001176 * d_HR + 1.97 * (1e-8) * d_HR ** 2
            eta = eta_sb * eta_trunc * eta_cos * eta_at * eta_ref
            eta_list.append(eta)
            eta_cos_list.append(eta_cos)
            eta_sb_list.append(eta_sb)
            eta_trunc_list.append(eta_trunc)
            print(eta_sb, eta_trunc)

        avg_eta = sum(eta_list) / len(eta_list)
        avg_eta_cos = sum(eta_cos_list) / len(eta_cos_list)
        avg_eta_sb = sum(eta_sb_list) / len(eta_sb_list)
        avg_eta_trunc = sum(eta_trunc_list) / len(eta_trunc_list)
        G0 = 1.366
        H = 3
        a = 0.4237 - 0.00821 * (6 - H)**2
        b = 0.5055 + 0.00595 * (6.5 - H)**2
        c = 0.2711 + 0.01858 * (2.5 - H)**2
        DNI = G0 * (a + b * np.exp(-c / sinalphas))
        E_divide_S = DNI * sum(eta_list) / len(x0)
        five_eta.append(avg_eta)
        five_cos.append(avg_eta_cos)
        five_sb.append(avg_eta_sb)
        five_trunc.append(avg_eta_trunc)
        five_EdivideS.append(E_divide_S)

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
workbook.save(filename="Result.xlsx")
