import numpy as np
from scipy.optimize import curve_fit
import openpyxl as op
import csv

d = 0.2767
lamb = 1.7725
m = 15
n = 40
r = 4
init = np.pi/2

I = []

for s in range(0, 180):
    with open("data1.csv", "r") as csvfile:
        reader = csv.reader(csvfile) # 创建一个reader对象
        tempI = np.array([float(row[s]) for row in reader]) # 用列表推导式获取第一列
    for i in range(0, 512):
        I.append(tempI[i])

k = range(0, 512 * 180)

kdata = np.array(k)
Idata = np.array(I)

def func(k, x0, y0, *theta):
    Ik = np.zeros_like(k)

    for i in range(0, 180):
        temp1 = ((k[k // 180 == i] % 180 - 255.5)*d + (x0 - 45)*np.sin(theta[i]) - y0 * np.cos(theta[i]))**2
        temp2 = m**2 * (np.sin(theta[i]))**2 + n**2 * (np.cos(theta[i]))**2
        temp3 = ((k[k // 180 == i] % 180 - 255.5)*d + x0*np.sin(theta[i]) - y0 * np.cos(theta[i]))**2
        res1 = 2*lamb*((np.clip(r**2 - temp1, 0, np.inf))**0.5)
        res2 = 2*lamb*m*n*(np.clip(temp2 - temp3, 0, np.inf))**0.5 / temp2
        Ik[k // 180 == i] = res1 + res2
    return Ik

xyt = [-9, 6]
for i in range(0, 180):
    xyt.append(119 + i)

popt, pcov = curve_fit(func, kdata, Idata, xyt)
print(popt)
