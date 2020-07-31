import numpy as np

noten = [100,4,2,4,5,235,5,52]
noten_np = np.array(noten,dtype = np.int32 )


print(noten)
print(noten_np.argmax())

zero_array = np.zeros(shape=10,dtype=np.int32)
zero_array

zeromatrix = np.zeros(shape=(10,3),dtype=np.int32)
zeromatrix

t_zeromatrix = np.reshape(zeromatrix,newshape=(3,10))
t_zeromatrix

o_matrix = np.transpose(t_zeromatrix)

(o_matrix == zeromatrix).all()