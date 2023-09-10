from sko.GA import GA
from tqdm import tqdm
import numpy as np
import random
x0 = []
y0 = []
c0 = []

dot_list = []
old_list = []

D = 0
ST = 10.5
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

xn = - cosalphas * singammas
yn = - cosalphas * cosgammas
zn = - sinalphas

G0 = 1.366
H = 3
a_d = 0.4237 - 0.00821 * (6 - H)**2
b_d = 0.5055 + 0.00595 * (6.5 - H)**2
c_d = 0.2711 + 0.01858 * (2.5 - H)**2
DNI = G0 * (a_d + b_d * np.exp(-c_d / sinalphas))
lambdai = [xn, yn, zn]


def generate_dotlist(yR, delta_r, a):
    global dot_list
    dot_list = []
    r = 108
    count = 0
    while r <= 350 + yR:
        N = int(np.floor((2 * np.pi * r) / (a + 5)))
        delta_theta = 2 * np.pi / N
        off = delta_theta * random.uniform(0, 1)
        for i in range(0, N):
            dot_list.append((r * np.cos(delta_theta * i + off),
                            r * np.sin(delta_theta * i + off) + yR,
                            count))
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
    global x0
    global y0
    global c0
    x0 = []
    y0 = []
    c0 = []
    for i in dot_list:
        x0.append(i[0])
        y0.append(i[1])
        c0.append(i[2])


def min_Fun(args):
    global old_list
    global dot_list
    yR, a, b, d, delta_r, delta_d, delta_b = args
    if (dot_list == []):
        generate_dotlist(yR, delta_r, a)
        old_list = [yR, a, b, d, delta_r, delta_d, delta_b]
    else:
        if old_list[0:7] != [yR, a, b, d, delta_r, delta_d, delta_b]:
            generate_dotlist(yR, delta_r, a)
            old_list[0:7] = [yR, a, b, d, delta_r, delta_d, delta_b]
    return len(dot_list) * a * b


def eq_Efield(args):
    global old_list
    global dot_list
    yR, a, b, d, delta_r, delta_d, delta_b = args
    if (dot_list == []):
        generate_dotlist(yR, delta_r, a)
        old_list = [yR, a, b, d, delta_r, delta_d, delta_b]
    else:
        if old_list[0:7] != [yR, a, b, d, delta_r, delta_d, delta_b]:
            generate_dotlist(yR, delta_r, a)
            old_list[0:7] = [yR, a, b, d, delta_r, delta_d, delta_b]
    global x0
    global y0
    mid = 80
    xR = 0
    eta_ref = 0.92
    mirror = {}
    for i in range(0, len(x0)):
        n = [x0[i] - xR, y0[i] - yR, d + delta_d * c0[i] - mid]
        n = n / np.linalg.norm(n)
        mirror[i] = n
    global lambdai

    n_dict = {}
    for i in range(0, len(x0)):
        n_dict[i] = - (mirror[i] + lambdai)
        n_dict[i] /= np.linalg.norm(n_dict[i])
    eta_list = []

    for i in range(0, len(x0)):
        eta_sb = 0.99
        cos_theta = - (lambdai[0] * n_dict[i][0] + lambdai[1] * n_dict[i][1] + lambdai[2]
                       * n_dict[i][2]) / (np.linalg.norm(lambdai) * np.linalg.norm(n_dict[i]))
        eta_cos = cos_theta
        d_HR = ((x0[i] - xR)**2 + (y0[i] - yR)**2 +
                (d + delta_d * c0[i] - mid)**2)**0.5
        eta_at = 0.99321 - 0.0001176 * d_HR + 1.97 * (1e-8) * d_HR ** 2
        eta = eta_sb * eta_cos * eta_at * eta_ref
        eta_list.append(eta)
    global DNI
    E_field = DNI * a * (b + delta_b * c0[i]) * sum(eta_list)
    return [E_field - 60000]


ga = GA(func=min_Fun, n_dim=7,    prob_mut=0.01,    size_pop=100,   max_iter=1000,
        lb=[-350, 6, 2, 2, 10, 0, -0.5], ub=[0, 8, 6, 6, 20, 0.5, 0], constraint_eq=[eq_Efield])
for i in tqdm(range(ga.max_iter), desc="Processing"):
    ga.run(1)
best_x, best_z = ga.run()
print('best_x,y:', best_x, '\n', 'best_z:', best_z)
