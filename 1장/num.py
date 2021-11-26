import numpy as np # numpy import 

x = np.array([1,2,3]) # 1차원 배열 선언
print(x)

a = np.array([1,2,3])
b = np.array([2,4,6])

print(a+b)
print(a-b)
print(a/b)

A = np.array([[1,2],[3,4]]) # 2차원 배열 선언
print(A)
print(A.shape) # 행과 열의 수 표현
print(A.dtype)

B = np.array([[3,0],[0,6]])
print(A+B)
print(A*B)

print(A * 10)

c = np.array([[1,2],[3,4]]) # numpy 로 행렬의 곱셈을 쉽게 할 수 있다.
d = np.array([10,20])
print(c*d)

x = np.array([[51,55],[14,19],[0,4]])
print(x)

x = x.flatten() # 2차원인 x를 1차원 배열로 변환
print(x)

print(x[np.array([0,2,4])]) # 인덱스가 0,2,4인 원소 얻기

print(x>15) # x의 원소 중 15 보다 큰 것만 bool type으로 출력

print(x[x>15])