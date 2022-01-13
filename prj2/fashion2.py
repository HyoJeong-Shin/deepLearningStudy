import tensorflow as tf
import numpy as np

# 쇼핑몰 의류이미지 자동분류 - cnn 적용

# 쇼핑몰 이미지 데이터셋 가져오기 
(trainX, trainY), (testX, testY)= tf.keras.datasets.fashion_mnist.load_data()   # 구글이 호스팅해주는 데이터셋 중 하나  

# 이미지 데이터 전처리
# 0~255를 input으로 넣는 것이 아닌 0~1로 미리 압축해서 넣음 -> 결과가 좋게 나올수도 있고, 처리 시간이 빨라질 수 있음
trainX = trainX / 255.0
testX = testX / 255.0


# numpy array자료의 shape 변경
trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1))   # 원래 shape [[0,0,...0]...[0,0,...0]...]  # reshape한 결과 [[[0],[0],...[0]]...[[0],[0],...[0]]]
testX = testX.reshape( (testX.shape[0], 28, 28, 1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot'] 


# 1. 모델 만들기
# 확률예측문제라면 마지막 레이어 노드수를 카테고리 개수만큼 생성하고, crossentropy라는 loss 함수 사용

# 모델 만들기 순서 : Conv - Pooling 여러번 함  (=> 다차원) 그리고 Flatten - Dense - 출력
model = tf.keras.Sequential([
    # 32개의 다른 feature(이미지) 생성  # (3,3) : kernel 가로세로 사이즈    # padding=same : 작아진 이미지 사이즈 이전과 똑같이 맞추기 위해 가로세로 1 pixel 더해줌     # relu쓰는 이유 : 이미지를 숫자로 바꾸면 0~255임. 이미지에는 음수값 올 수 없기에 음수를 0으로 바꿔주는 relu함수 씀     # input_shape : 데이터 하나의 shape으로 Conv2D는 4차원 데이터 입력 필요 ex) (60000, 28, 28, 1) (60000, 28, 28, 3)
    tf.keras.layers.Conv2D( 32, (3,3), padding="same", activation='relu', input_shape=(28, 28, 1) ),   # color-image일 경우 input_shape (28, 28, 3)  # 컬러이미지는 [[[0,0,0],[0,0,0]...[0,0,0]][[0,0,0],[0,0,0]...[0,0,0]]] 이런식으로 되어있음
    tf.keras.layers.MaxPooling2D( (2,2) ),      # 사이즈를 줄여주고, 중요한 정보들을 가운데로 모음  # (2,2) : pooling size
    #tf.keras.layers.Dense(128, input_shape=(28, 28), activation="relu"),      # relu : 음수를 다 0으로 만듦 => convolution layer에서 자주 사용
    tf.keras.layers.Flatten(),                          # 다차원 행렬을 1차원으로 압축해주는 Flatten 레이어
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")     # sigmoid : 결과를 0~1로 압축 => binary 예측문제에 사용(대학원 붙는다/안붙는다), 마지막 노드 갯수는 1개   # softmax : 결과를 0~1로 압축 => 카테고리 예측문제에 사용, 예측한 10개 확률을 다 더하면 1나옴
])

# 모델 아웃라인 출력해보기 : 모델 잘짰는지 확인
model.summary()     # input_shape=()를 넣어줘야 summary 보기 가능   # input_shape = 데이터 하나의 shape

# 2. 모델 컴파일
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# 3. 모델 fit하기 (학습)
# overfitting 일어나는 시점 확인 가능 : epoch 1회 끝날 때마다 채점 (주기적으로 테스트)  => 일반적인 학습 방법
# val_accuracy 높일 방법 찾기 (dense layer 추가? conv+pooling 추가?)
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)

# 4. 학습 후 모델 평가 (학습용 데이터가 아닌 컴퓨터가 처음 보는 테스트 데이터 넣어야 함)
#score = model.evaluate( testX, testY)
#print(score)    # [loss, accuracy] 출력됨   # fit값과 평가값의 loss, accuracy가 조금 다름 => overfitting 현상 : training dataset을 외워 새로운 data를 넣었을 때 못 푸는 현상
