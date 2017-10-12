import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt

mu1 = np.array([[1, 1]])
mu2 = np.array([[4, 4]])
mu3 = np.array([[8, 1]])
Sigma = np.array([[2, 0], [0, 2]])
R = cholesky(Sigma)
a = np.dot(np.random.randn(333, 2), R) + mu1
b = np.dot(np.random.randn(333, 2), R) + mu2
c = np.dot(np.random.randn(334, 2), R) + mu3

s = np.append(a,b,0)
s = np.append(s,c,0)

"""
print np.mean(a,0)
print np.var(a[:,0])
print np.var(a[:,1])
print np.mean(b,0)
print np.var(b[:,1])
print np.var(b[:,1])
print np.mean(c,0)
print np.var(c[:,1])
print np.var(c[:,1])
"""
print np.mean(s,0)
file_object = open('output.txt', 'w')
file_object.write(str(np.cov(s)))
file_object.close()
#print np.cov(s)

plt.plot(s[:,0],s[:,1],'o', alpha=0.6)
plt.show()