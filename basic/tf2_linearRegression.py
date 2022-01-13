import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 딥러닝으로 간단한 수학문제 풀어보기 (Linear Regression)
# 키와 신발사이즈는 어떤 관련이 있을까?
키 = [170, 180, 175, 160]
신발 = [260, 270, 265, 255]

# y = ax + b
# y : 신발, x : 키    키를 대입하면 신발 사이즈를 알 수 있도록 a와 b를 구하면 되는 문제    신발 = 키 * a + b

# 키로 신발사이즈를 추론해보자
키 = 170
신발 = 260

# 초기값 아무거나 설정한 후, a,b를 좋은 결과 나올 때까지 학습시킬 것
a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    예측값 = 키 * a + b
    # return (손실값)^2         손실값(=오차 =실제값 - 예측값)
    # 제곱 함수 : tf.square
    return tf.square(260 - 예측값)

# 경사하강법 이용해서 변수들을 업데이트 해주는 함수 : tf.keras.optimizers
# gradient를 알아서 스마트하게 바꿔줌  : Adam
# learning_rate : 얼마만큼 변수들을 업데이트 해줄지 
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# 경사하강법 실행  opt.minimize(손실함수, 경사하강법으로 업데이트할 weight Variable 목록)
opt.minimize(손실함수, var_list=[a,b])  # 경사하강1번 해줌 = a, b를 1번 수정해줌 (실제값과 유사한 a, b로 수정)

# 여러번 반복 (최저값이 나올 때까지 반복)
for i in range(300):
    opt.minimize(손실함수, var_list=[a,b])
    print(a.numpy(), b.numpy())