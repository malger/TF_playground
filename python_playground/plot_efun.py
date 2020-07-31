import numpy as np
import matplotlib.pyplot as plt


def e_function(a, b):
    return np.exp(np.arange(a,b+1))

a = 1
b = 5
e_list = e_function(a,b)

plt.plot(np.arange(a,b+1),e_list)
plt.xlabel("x")
plt.ylabel("e(x)")

plt.show()

