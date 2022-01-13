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
    print(rows['gre'])

exit()


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

# 2. model compile하기
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 0과 1사이 분류/확률문제에 쓰는 loss함수 : binary_crossentropy

# 3. model 학습(fit) 시키기
# x에는 학습데이터, y는 실제정답데이터, epochs : 몇번 학습을 시킬지
model.fit( x데이터, y데이터, epochs=10)

# inputData : x = [ [데이터1], [데이터2], [데이터3] ...]    ex) [[380, 3.21, 3], [660, 3.67, 3], [], [] ...]
# label(답안) : y = [정답1, 정답2, 정답3 ...]   ex) [0, 1, 1, 1 ...]