import tensorflow as tf
import matplotlib.pyplot as plt

# 쇼핑몰 의류이미지 자동분류

# 쇼핑몰 이미지 데이터셋 가져오기 
(trainX, trainY), (testX, testY)= tf.keras.datasets.fashion_mnist.load_data()   # 구글이 호스팅해주는 데이터셋 중 하나  

# trainX=이미지6만개 
#print(trainX[0])       # 첫째 이미지
#print(trainX.shape)    # 출력값 : (60000, 28, 28) => [28개의 숫자가 들어있는 리스트]가 28개 있음 * 6만 (그게 6만개 존재)

# trainY=정답들어있는리스트(=label)
#print(trainY)
# 카테고리 수 (label수)     # 이 사진은 어떤 카테고리에 속할 확률이 높을까요 (마지막layer에 softmax써야함)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot'] 

# 해당하는 이미지 띄우기
plt.imshow( trainX[0] )
plt.gray()      # 흑백으로 출력하기
plt.colorbar()  # 어떤 색상인지 수치화해서 보여줌
plt.show()


# 1. 모델 만들기
# 확률예측문제라면 마지막 레이어 노드수를 카테고리 개수만큼 생성하고, crossentropy라는 loss 함수 사용
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28, 28), activation="relu"),      # relu : 음수를 다 0으로 만듦 => convolution layer에서 자주 사용
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Flatten(),                          # 다차원 행렬을 1차원으로 압축해주는 Flatten 레이어
    tf.keras.layers.Dense(10, activation="softmax")     # sigmoid : 결과를 0~1로 압축 => binary 예측문제에 사용(대학원 붙는다/안붙는다), 마지막 노드 갯수는 1개   # softmax : 결과를 0~1로 압축 => 카테고리 예측문제에 사용, 예측한 10개 확률을 다 더하면 1나옴
])

# 모델 아웃라인 출력해보기 : 모델 잘짰는지 확인
model.summary()     # input_shape=()를 넣어줘야 summary 보기 가능   # input_shape = 데이터 하나의 shape

# 2. 모델 컴파일
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# 3. 모델 fit하기 (학습)
model.fit(trainX, trainY, epochs=5)
