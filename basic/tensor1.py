import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

텐서 = tf.constant( [3,4,5] )
텐서2 = tf.constant( [6,7,8])

print(tf.add(텐서, 텐서2))
print(tf.subtract(텐서, 텐서2))
print(tf.divide(텐서, 텐서2))
print(tf.multiply(텐서, 텐서2))

텐서3 = tf.constant( [ [1,2],
                      [3,4] ])
텐서4 = tf.constant( [ [1,5],
                      [3,7] ])

# 행렬의 곱 (dot product)
print(tf.matmul(텐서3, 텐서4))


# 데이터에 뭘 담을지 모를 때 : 0만 담긴 텐서 만들어줌
텐서4 = tf.zeros( [2,2,3] )
print(텐서4)

# shape : 몇차원 데이터인지 보여줌
print(텐서.shape)
print(텐서3.shape)  # 2행 2열 데이터 (2개의 데이터가 담긴 리스트가 2개 존재)

텐서5 = tf.constant( [ [1,2,3],
                      [4,5,6] ], tf.float32)
print(텐서5.shape)  # 2행 3열 데이터 (3개의 데이터가 담긴 리스트가 2개 존재)  => 뒤에서부터 해석


# 텐서의 자료형 dtype(data type)   정수 : int    실수 : float
print(텐서)
# 텐서 자료형 변형
print(텐서5)
print(tf.cast(텐서5, tf.int32))


# weight 저장하고 싶으면 Variable(변수) 만들기    => 변경이 쉽게 됨
# 고정된 값 : tf.constant  # 변경 가능한 값 : tf.Variable
w = tf.Variable(1.0)
print(w)

# 변수에 저장된 값 불러옴
print(w.numpy())

# 변수 변경
w.assign(2)
print(w)