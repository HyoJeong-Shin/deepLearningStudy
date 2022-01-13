import tensorflow as tf
# 대학원 붙을 확률 계산

# 1. 딥러닝 모델 디자인
# Sequential : 신경망 레이어 만들어줌
# node개수는 결과가 잘 나올 때까지 실험으로 파악해야 함 (관습적으로 2의 제곱수로 표현)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),    # 일반적인 딥러닝 hidden layer  # layer1  # Dense안에 들어간 숫자 : node의 개수  # 두번째 parameter : 활성함수(activation function)
    tf.keras.layers.Dense(128, activation='tanh'),   # layer2
    tf.keras.layers.Dense(1, activation='sigmoid')      # layer3      # 마지막 출력 레이어  # 예측결과가 0~1사이의 확률이려면 활성함수 sigmoid로 설정
])
