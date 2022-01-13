카테고리 확률 예측 문제

1. 마지막 레이어 노드 수를 카테고리 개수만큼 생성
   
2. 마지막 레이어의 activation function은 softmax 사용
   - sigmoid vs softmax
   - sigmoid : 결과를 0~1로 압축 => binary 예측문제에 사용(대학원 붙는다/안붙는다), 마지막 노드 갯수는 1개
   - softmax : 결과를 0~1로 압축 => 카테고리 예측문제에 사용, 예측한 10개 확률을 다 더하면 1 나옴

3. crossentropy라는 loss 함수 사용
   1. trainY가 원핫인코딩이 되어있을 때 categorical_crossentropy 사용
   2. trainY가 0, 1, 2, ... 등 정수로 되어있을 때 sparse_categorical_crossentropy 사용

- 원핫인코딩이란?
  - 4개의 카테고리 중 어디에 들어갈지 예측하는 문제
    1. 정답(label)이 정수로 표현되어있는 경우  [1, 2, 0, 1, 3 ..] loss함수 sparse_categorical_crossentropy 사용
    2. 정답을 첫째 카테고리에 속하면 [1, 0, 0, 0], 둘째 카테고리에 속하면 [0, 1, 0, 0], 셋째 카테고리에 속하면 [0, 0, 1, 0] 등 이렇게 만들면 원핫인코딩
        - [1, 3, 0, 2]라는 정답데이터를 원핫인코딩하면, [ [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0] ] 이렇게 표현가능
        - loss함수 categorical_crossentropy 사용


Convolutional layer

- 이미지 데이터를 flatten하는 것의 문제점
  - tf.keras.layers.Flatten() : 다차원 행렬을 1차원으로 압축해주는 Flatten 레이어
  - 이미지를 해체해서 딥러닝을 돌리는 것으로 예측모델의 응용력 사라짐
  
- 해결책 (feature extraction)
  1. 이미지에서 중요한 정보 추려서 복사본 여러개 만들기 (새로운 이미지 레이어를 여러개 만듦)
  2. 복사본엔 이미지의 중요한 feature(특성)이 담김  ex) 동그란 부분 강조, 특정 색상 강조 등 (이미지의 특성들이 각각 다르게 강조되게)
  3. 이걸로 학습

- feature map 만들기
  - 입력 이미지에서 중요한 정보만 뽑아 새로운 이미지 layer 만드는 작업
  - kernel을 거쳐 새로운 이미지 만듦
  - tensorflow는 여러가지 kernel들을 자동적용해서 layer 만들어줌

- 단순 convolution의 문제점 : feature 위치
  - ex) 자동차 이미지 : 바퀴 인식을 땅에 동그란 부분이 있으면 바퀴라고 학습 -> 자동차의 위치가 달라지면 바퀴 인식 못함 (응용력X)
  - 해결책 : Pooling layer (Down sampling)
    - 이미지 축소시킴. 이미지 size만 줄이는 것이 아닌, 이미지의 중요한 부분은 유지한채 가운데로 옮김
    - 여러가지 방법 존재
      - Max Pooling : 최댓값만 추리는 방법, 가장 많이 쓰는 방법
      - Average Pooling : 평균값으로 추림
    - 장점 ( translation invariance : 이미지가 어디있든 잘 인식함. 이미지 위치에 영향 받지 않음 )
      - Convolutional + Pooling layer 도입시 특징추출 + 특징을 가운데로 모아줌

- CNN (Convolutional Neural Network) 일반적인 구성법
  - input과 Neural Network 사이에 Convolutional layer + Pooling layer 적용
  - filter는 keras가 알아서 구성 가능