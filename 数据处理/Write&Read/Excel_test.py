import pandas as pd
import openpyxl as op

data = pd.read_excel('test.xlsx', sheet_name= 'Sheet1')

name_init = []

# 自动跳过表头
for row in data.iloc:
    name_init.append(row[0])

# print(name_init)


# 在这里才对数据行进行筛选
name = name_init[1:]
print(name)
# 即得列表name




# 接下来把c0写到result.xlsx中
workbook = op.Workbook()
worksheet1 = workbook.active
worksheet1.title = "Sheet1"
worksheet2 = workbook.create_sheet(title="Sheet2")

# 新建表头（二维列表）
data1 = [["Apex", "Legends"]]
data2 = [["Genshin", "Impact"]]

# 按行添加元素
data1.append([0, 0])
data1.append([1, 1])
data2.append([2, 2])
data2.append([3, 3])

# 添加到worksheet中
for row in data1:
    worksheet1.append(row)
for row in data2:
    worksheet2.append(row)

workbook.save(filename="result.xlsx")