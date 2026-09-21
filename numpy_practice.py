import numpy as np
#import pandas as pd
numbers = np.array([10,20,30,40,50])
print(numbers)
print(numbers*2)
print([10,20]*2)
print(numbers[0])
print(numbers[-1])
print(numbers[0:3])
print(numbers[2:])
print(numbers[-4:-1])
print(numbers.ndim)
matrix=np.array([
    [10,20,30],
    [40,50,60]
])
print(matrix)
print(matrix.ndim)
print(matrix.shape)
matrix2=np.array([
    [1,2],
    [3,4],
    [5,6]
])
print(matrix2)
print(matrix2.ndim)
print(matrix2.shape)
print(matrix[0,1])
numbers= np.array([10, 20, 30, 40, 50, 60])
print(numbers.min())
print(numbers.max())
print(numbers[numbers>30]) 
print(numbers[numbers<30]) 
print(numbers>30) 
print(numbers.shape)
print(numbers.reshape(2,3))
print(numbers.reshape(3,2))
a=np.array([10,20,30])
b=np.array([1,2,3])
print(a+b)
print(a*b)
print(a-b)
print(np.dot(a,b))
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

add_values = np.array([1, 2, 3])

print(matrix + add_values)