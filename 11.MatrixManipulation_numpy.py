import numpy as np
A=np.array([[1,2,3],[3,4,5]])
B=np.array([[4,6,8],[3,3,6]])

print(A+B)

print(np.dot(A,B))

print(A.T)
print(A.flatten())
print(A.reshape(3,2))
print(A.ravel())
print(A.T.flatten())
