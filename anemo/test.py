import numpy as np
import pylab as plt
from scipy import sparse
import pandas as pd

T0 = 75 + 273.15
T5 = 37 + 273.15

def tpr(k, t):
    return T0 - (T0-T5)* np.exp(-k * t)


k = 113

T0T = []
for t in range(0, 5401):
    T0T.append(tpr(k, t))


print(int(15.2 / 0.01)  + 1)