import sys 
import os
import numpy as np
import pickle # 파이썬에서 리스트나 클래스 이외의 자료형을 파일로 저장하고자 할때 사용
from common.functions import sigmoid,softmax


sys.path.append(os.pardir) # 부모 디렉토리
from dataset.mnist import load_mnist # dataset폴더에 있는 mnist라는 파일에서 load_mnist라는 함수를 import 해라
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환하는 함수, uint8 : 2^8만큼 표현가능 0~255
    pil_img.show()

def get_data():    
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False, one_hot_label=False) # flattern = True로 설정해 읽어 들인 이미지는 1차원 넘파이 배열로 저장
    return x_test, t_test

def init_network():
    with open("C:\image\sample_weight.pkl",'rb') as f: # sample_weight-이미학습된 개체, 왜인지몰라도 같은 폴더에 넣어놨더니 못읽어오길래 다른 경로로 집어넣어서 실행함. open_as rb는 읽기모드와 바이너리모드가 동시에 적용된 것.
        # sample_weight를 불러와서 f라는 변수로 치환
        network = pickle.load(f)
    return network

def predict(network, x): # 각 레이블의 확률을 넘파이 배열로 반환한다. 
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    
    return y
'''
x, t = get_data()
network = init_network()
accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]: # 신경망이 숫자를 맞췄다면
        accuracy_cnt += 1 # 그 횟수를 증가시킨다.
'''
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x), batch_size): # 0부터 batch_size 만큼 묶어나간다.
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch,axis = 1) # 100x10의 배열 중 1차원을 구성하는 각 원소에서(1번째 차원을 축으로) 최대값의 인덱스를 찾도록 한 것
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
print('Accuracy:' + str(float(accuracy_cnt)/len(x))) # len(x)는 전체 이미지 숫자이고 신경망이 맞춘 숫자를 전체 이미지 숫자로 나눠서 정확도를 구한다. 
