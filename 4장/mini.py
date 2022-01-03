import sys,os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist # MNIST 데이터 셋을 읽어오는 함수(훈련 데이터와 시험 데이터를 읽는다)

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True) # one_hot_label = True 로 원-핫 인코딩을 한다. -> 정답위치만 1 나머지 0

print(x_train.shape) # (60000,784)
print(t_train.shape) # (60000,10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 훈련 데이터 중 10개만 랜덤으로 추출하는 함수
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(np.random.choice(60000,10))

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size) # 정답 레이블
        y = y.reshape(1,y.size) # 신경망의 출력
    
    batch_size = y.shape[0] 
    return -np.sum(t*np.log(y+1e-7)) / batch_size