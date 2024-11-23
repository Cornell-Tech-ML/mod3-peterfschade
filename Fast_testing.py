
import minitorch
from minitorch import MathTestVariable, Tensor, TensorBackend, grad_check
import numpy as np
from numba import int32, re

simpleTensorBackend = minitorch.TensorBackend(minitorch.SimpleOps)
fastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)

size_a = 2
size_b = 2
size_in = 2
x1 = [
            [[np.random.uniform(0, 2) for i in range(size_in)] for j in range(size_a)]
            for _ in range(4)
        ]
y1 = [
            [[np.random.uniform(0, 2) for i in range(size_b)] for j in range(size_in)]
            for _ in range(4)
        ]

print(np.matmul(x1, y1))
print(np.matmul(x1, y1).shape)

A = minitorch.tensor(x1, backend=fastTensorBackend)
B = minitorch.tensor(y1, backend=fastTensorBackend)
print('-------------------------')
Z = A.__matmul__(B) 
print(Z.shape)

print(Z)
#print('-------------------------')
C = minitorch.tensor(x1, backend=simpleTensorBackend)
D = minitorch.tensor(y1, backend=simpleTensorBackend)
#print("here")
Z2 = C.__matmul__(D)
print(Z2)


#A = Tensor(minitorch.TensorData([1,2,3,4], shape=(1,4)), backend=FastTensorBackend)
#B = Tensor(minitorch.TensorData([2,2], shape=(2,1)), backend=FastTensorBackend)
#A.requires_grad_(True)
#B.requires_grad_(True)

#Z = A.__matmul__(B)
#print(Z)
#print('-------------------------')

#print('-------------------------')
#print(Z1.shape)