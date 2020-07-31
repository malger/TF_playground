import matplotlib.pyplot as plt

# list comprehensions
l = [x+1 for x in range(10)]
l

#matrix 
m = [[1,2],
     [3,4]]
m

m2 = [[i+j for j in range(3)] for i in range(4)]
m2

x1 = m2[0]
x2 = m2[1]
x3 = m2[2]

g = [range(2)]
g

plt.scatter(x1,x2,x3)
