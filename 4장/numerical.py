import sys
import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f,x):
    h = 0.0001
    return (f(x+h)-f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f,x):
    h = 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열 형성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
    return grad

def gradient_descent(f,init_x, lr=0.01, step_num = 100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad
    
    return x

x = np.arange(0.0,20.0,0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

print(numerical_diff(function_1,5))
print(numerical_diff(function_1,10))
print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([0.0,2.0])))
print(numerical_gradient(function_2,np.array([3.0,0.0])))

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))
