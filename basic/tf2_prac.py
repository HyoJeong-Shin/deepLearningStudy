import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# x와 y의 관계 : x에 2 곱하고 1 더하면 y 나옴
train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

# 변수 안에 들어갈 값 randomize (무작위로 집어넣음)
a = tf.Variable(0.1)
b = tf.Variable(0.1)

# 1. 모델 만들기

def 손실함수(a,b):
    예측_y = train_x * a + b
    # mean squared error 함수 (평균 제곱): 정수 예측   => 이 코드에서는 이것 사용
    # cross entropy 함수 : 카테고리 분류, 확률 예측
    # tf.keras.losses.mse(실제값List, 예측값List) : 한 번에 다 loss값 계산해줌
    return tf.keras.losses.mse(train_y, 예측_y)


# 2. 학습하기 (최적화하기) (=경사하강 시키기)
opt = tf.keras.optimizers.Adam(learning_rate=0.01)     # learning_rate : 결과 잘나올 때까지 수정 필요

for i in range(2900):
    opt.minimize(lambda:손실함수(a,b), var_list=[a,b])
    print(a.numpy(), b.numpy())
