import pandas as pd
import csv

# usecols可以选取需要的列
data = pd.read_csv('test.csv', usecols=range(0, 2), encoding="gbk", dtype=float)

name_init = []

# 自动跳过表头
for row in data.iloc:
    name_init.append(row[0])

# print(name_init)


# 在这里才对数据行进行筛选
name = name_init[1:]
print(name)
# 即得列表name



# 生成result.csv文件
with open('result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    data = [["Apex", "Legends"]]
    data.append([0, 0])

    writer.writerows(data)