import pandas as pd
import pylab as plt
data = pd.read_excel('result1.xlsx', sheet_name= 'Sheet1')


a = []
b = []
c = []
d = []
e = []
for row in data.iloc:
    a.append(row[0])
    b.append(row[1])
    c.append(row[2])
    d.append(row[3])
    e.append(row[4])

t = range(1, 13)

plt.rc('text', usetex=True)
plt.rc('font', size = 10)
plt.xticks(range(1, 13))
plt.plot(t, a, label= "$ \eta $")
plt.plot(t, b, label= "$ \eta_{\mathrm{cos}} $")
plt.plot(t, c, label= "$ \eta_{\mathrm{sb}} $")
plt.plot(t, d, label= "$ \eta_{\mathrm{trunc}} $")
plt.plot(t, e, label= "$ E_{field} / S $")
plt.legend(bbox_to_anchor=(0.75, 0.9), loc='upper left', borderaxespad=0)
plt.show()
