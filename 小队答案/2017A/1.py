import numpy as np
from scipy.optimize import leastsq
import openpyxl as op
import csv

R = []

d = 0.2767
lamb = 1.7725
m = 15
n = 40
r = 4
init = np.pi/2

for s in range(0, 180):
    with open("data1.csv", "r") as csvfile:
        reader = csv.reader(csvfile) # 创建一个reader对象
        I = np.array([float(row[s]) for row in reader]) # 用列表推导式获取第一列


    theta = []
    k = np.array(range(1,513))


    def fun(x):
        temp1 = ((k - 256.5)*d + (-9.2696 - 45)*np.sin(x[0]) - 6.2738 * np.cos(x[0]))**2
        temp2 = m**2 * (np.sin(x[0]))**2 + n**2 * (np.cos(x[0]))**2
        temp3 = ((k - 256.5)*d + (-9.2696)*np.sin(x[0]) - 6.2738 * np.cos(x[0]))**2
        res1 = 2*lamb*((np.clip(r**2 - temp1, 0, np.inf))**0.5)
        res2 = 2*lamb*m*n*(np.clip(temp2 - temp3, 0, np.inf))**0.5 / temp2
        return (res1 + res2) - I

    x0 = np.array([init])

    res = leastsq(fun, x0)
    init = res[0][0]
    R.append(np.rad2deg(res[0][0]))



workbook = op.Workbook()
worksheet1 = workbook.active
worksheet1.title = "theta"
data1 = []

for c in range(0, 180):
    data1.append([R[c]])
    
for row in data1:
    worksheet1.append(row)

workbook.save(filename="theta.xlsx")