# Rx = p
# xj是要求的第j格吸收率，
# R(i,j)表示，第i = 512 * （t-1） + k条是否经过第j格
# pi是第i = 512 * （t-1） + k条的I/lambda 
import matplotlib.pyplot as plt
import numpy as np
import csv
import openpyxl as op
lamb = 1.7725
d = 0.2767
tau = 100/256
x0 = -9.2696
y0 = 6.2738
p = []  # p[theta-1][k-1] 代表theta , k处的线积分
for s in range(0, 180):
    with open("data3.csv", "r") as csvfile:
        reader = csv.reader(csvfile) # 创建一个reader对象
        I = [(float(row[s]) / lamb) for row in reader] # 用列表推导式获取第一列
    for i in range(0, 512):
        p.append(I[i])

# theta[t-1] 代表角度t（角度制

with open("theta.csv", "r") as csvfile:
    reader = csv.reader(csvfile) # 创建一个reader对象
    theta = [float(row[0]) for row in reader] # 用列表推导式获取第一列


#  求R阵(180*512, 256*256)

R = []
for t in range(0,180):
    for k in range(0,512):
        print(512 * t + k)
        tempr = np.zeros((256*256),dtype= np.uint8)
        for m in range(0,256):
            if abs(np.tan(theta[t])) < 1:
                x = -50 + tau * (2*m + 1)/2
                ya = ((x - x0) * np.sin(theta[t]) - (k - 256)*d)/np.cos(theta[t]) + y0
                yb = ((x - x0) * np.sin(theta[t]) - (k - 257)*d)/np.cos(theta[t]) + y0
                if ((ya > 50 - tau/2 and yb > 50 - tau/2) or (ya < -(50 - tau/2) and yb < -(50 - tau/2))):
                    continue
                ya_num = (ya + 50) / tau - 0.5
                yb_num = (yb + 50) / tau - 0.5
                a = int(np.ceil(min(ya_num ,yb_num)))
                if (a < 0):
                    a = 0
                b = int(np.floor(max(ya_num, yb_num)))
                if (b > 255):
                    b = 255
                for _ in range(a,b+1):
                    tempr[256*m+_] = 1
            else:
                y = -50 + tau * (2*m + 1)/2
                xa = ((y - y0) * np.cos(theta[t]) + (k - 256)*d)/np.sin(theta[t]) + x0
                xb = ((y - y0) * np.cos(theta[t]) + (k - 257)*d)/np.sin(theta[t]) + x0
                if ((xa > 50 - tau/2 and xb > 50 - tau/2) or (xa < -(50 - tau/2) and xb < -(50 - tau/2))):
                    continue
                xa_num = (xa + 50) / tau - 0.5
                xb_num = (xb + 50) / tau - 0.5
                a = int(np.ceil(min(xa_num ,xb_num)))
                if (a < 0):
                    a = 0
                b = int(np.floor(max(xa_num, xb_num)))
                if (b > 255):
                    b = 255
                for _ in range(a,b+1):
                    tempr[256*_+m] = 1
        R.append(tempr)

x = np.zeros(256 * 256)


# 进入迭代：3I - 8I
for i in range(0, 180*512 * 8):
    print(i)
    m = i % (180 * 512)
    if np.dot(R[m],R[m]) ==0 :
        continue
    A = 1 * (p[m] - np.dot(R[m], x)) / np.dot(R[m],R[m])
    x = x + A * R[m]


fig = plt.figure()
ax3 = plt.axes(projection='3d')

jj = []
aa = []

for m in range(0, 256):
    jj.append(m)
    aa.append(m)

J, A = np.meshgrid(jj, aa)
list_out = []
for aa_i in aa:
    list_in = []
    for jj_i in jj:
        list_in.append(x[jj_i+ aa_i * 256])
    list_out.append(list_in)
P = np.array(list_out)

ax3.plot_surface(J, A, P, cmap='rainbow')
plt.show()



workbook = op.Workbook()
worksheet1 = workbook.active
worksheet1.title = "success"
data1 = []

for c in range(0, 256):
    x1 = []
    for e in range(0, 256):
        x1.append(x[256*c + e])
    data1.append(x1)
    
for row in data1:
    worksheet1.append(row)

workbook.save(filename="success.xlsx")