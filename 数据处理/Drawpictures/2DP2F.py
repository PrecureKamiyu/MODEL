import numpy as np
import pylab as plt

def func(x):
    return x ** 4 + x ** 3 + x ** 2 + x

x = np.linspace(0, 1, 11)
y = func(x)
plt.plot(x, y, label= "crookedline")
plt.plot(x, y, 'o')
plt.legend()
plt.show()
