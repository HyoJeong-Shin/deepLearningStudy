import pandas as pd

# dataframe : 열과 행이 있는 데이터
data = pd.read_csv('gpascore.csv')
print(data)


# 데이터 전처리하기

# 유용한 pandas 사용법
#print(data.isnull().sum())     # 빈칸인 데이터 세주기
# data = data.fillna(100)   # 빈칸을 100이라는 숫자로 채워줌
# print(data['gpa'])        # 원하는 열만 출력  (gpa열 출력)
# print(data['gpa'].min())  # 최솟값    # 최댓값 : max()      # 해당 열의 데이터 개수 세기 : count()
# exit()  # 이 위까지만 코드 실행

data = data.dropna()        # NaN/빈값있는 행 제거

# y데이터 = admit에 있는 데이터 정렬    # values : 데이터들을 리스트로 담음
y데이터 = data['admit'].values


x데이터 = []

# i : 행번호 rows : 각 행의 데이터들  iterrows() : dataframe의 data를 한 행씩 출력
for i, rows in data.iterrows():
    x데이터.append([ rows['gre'], rows['gpa'], rows['rank'] ])
    

import numpy as np
import tensorflow as tf
# 대학원 붙을 확률 계산  1. 모델 만들고 -> 2. 데이터 집어넣고 학습 -> 3. 새로운 데이터 예측

# 1. 딥러닝 모델 디자인
# Sequential : 신경망 레이어 만들어줌
# node개수는 결과가 잘 나올 때까지 실험으로 파악해야 함 (관습적으로 2의 제곱수로 표현)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),    # 일반적인 딥러닝 hidden layer  # layer1  # Dense안에 들어간 숫자 : node의 개수  # 두번째 parameter : 활성함수(activation function)
    tf.keras.layers.Dense(128, activation='tanh'),   # layer2
    tf.keras.layers.Dense(1, activation='sigmoid')      # layer3      # 마지막 출력 레이어  # 예측결과가 0~1사이의 확률이려면 활성함수 sigmoid로 설정
])

# 2. model compile하기
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 0과 1사이 분류/확률문제에 쓰는 loss함수 : binary_crossentropy

# 3. model 학습(fit) 시키기  =  w값 최적화
# x에는 학습데이터, y는 실제정답데이터, epochs : 몇번 학습을 시킬지
model.fit( np.array(x데이터), np.array(y데이터), epochs=1000)     # 데이터를 list 그대로가 아닌, numpy array 또는 tf tensor로 변환해야 학습 가능

# inputData : x = [ [데이터1], [데이터2], [데이터3] ...]    ex) [[380, 3.21, 3], [660, 3.67, 3], [], [] ...]
# label(답안) : y = [정답1, 정답2, 정답3 ...]   ex) [0, 1, 1, 1 ...]

# 학습시킨 후 나오는 결과값 설명
# loss : 예측값과 실제값의 차이값(손실값) => 적어질 수록 학습 잘 되고 있는 것
# accuracy : 예측값이 실제값과 얼마나 정확히 맞는지 평가 => 높을수록 좋음


# 4. 학습시킨 모델로 예측하기
# gre성적 750, 학점3.7, rank4인 사람과 400, 2.2, 1인 사람의 대학원 합격확률 예측
예측값 = model.predict( [ [750, 3.7, 4], [400, 2.2, 1]])
print(예측값)       # 출력값 : [[0.85545945] [0.05677262]] => 85%, 5% 확률로 예측함