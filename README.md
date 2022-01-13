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
  1. 정답(label)이 정수로 표현되어있는 경우  '''[1, 2, 0, 1, 3 ..]''' loss함수 sparse_categorical_crossentropy 사용
  2. 정답을 첫째 카테고리에 속하면 [1, 0, 0, 0], 둘째 카테고리에 속하면 [0, 1, 0, 0], 셋째 카테고리에 속하면 [0, 0, 1, 0] 등 이렇게 만들면 원핫인코딩
     - [1, 3, 0, 2]라는 정답데이터를 원핫인코딩하면,  [ [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0] ] 이렇게 표현가능
     - loss함수 ategorical_crossentropy 사용