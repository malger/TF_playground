import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand



noten = [100,4,2,4,5,235,5,52]
noten_np = np.array(noten,dtype = np.int32 )
sortiert = noten_np.copy()
sortiert.sort()

plt.plot([1,2,3,4,5,6,7,8],noten_np,color="blue")
plt.plot([1,2,3,4,5,6,7,8],sortiert,color="red")

plt.legend(["random","ordered"])
plt.title("super geile grafik")
plt.show();
