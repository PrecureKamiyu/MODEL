import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

k4 = 0.028
us = 37 + 273.15

data1 = pd.read_excel('dTdx_persec.xlsx')
first_column = data1.iloc[0:5401, 0]

print(first_column)
dTdx_list = [0]
for i in range(0, 5400):
    dTdx_list.append(first_column[i] + 273.15)

data2 = pd.read_excel('CUMCM-2018-Problem-A-Chinese-Appendix.xlsx', sheet_name='附件2')
first_column = data2.iloc[1:, 1]

temp_list = []
for i in range(1, 5402):
    temp_list.append(first_column[i] + 273.15)

k_list = []

for i in range(0, 5401):
    if(us == temp_list[i]):
        continue
    else:
        k_list.append(k4 * dTdx_list[i] / (us - temp_list[i]))

print(k_list)

def rms(numbers):
    return sum(numbers[30:])/len(numbers[30:])

print(rms(k_list))

# ks = 8.186