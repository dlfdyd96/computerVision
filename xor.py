import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# 학습 데이터 구성은 2차원 행렬로 표현
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

# 하이퍼 파라미터를 찾는 과정 중에는 다음과 같이 랜덤 씨드를 고정이 좋음
# - 하이퍼 파라미터 : 히든 노드의 갯수, Learning-rate => 실험할 사람이 조정할 값 들
# - numpy와 tnesorflow 모두에 적용
np.random.seed(123)
tf.set_random_seed(123)

learning_rate = 0.01

# 제일먼저 할일 x-data를 담을 x-tensor를 만드는 것
x = tf.placeholder(tf.float32, shape=(None, 2)) # shape=(4, 2) None하면 자동으로 설정됨
y = tf.placeholder(tf.float32, shape=(None, 1))

# Weight 설정
# hidden Node의 weight : 입력2개, hidden Layer의 Node(출력)가 3개
# 초기화
h1_w23 = tf.truncated_normal(shape=(2, 3), mean=0, stddev=1.0, dtype=tf.float32) # stddev : 표준편차
h1_b1 = tf.zeros(shape=(1,3)) # bias(절편) 은 보통 zeros로 초기화함
# 변수만듬
w1 = tf.Variable(h1_w23)
b1 = tf.Variable(h1_b1)

# hidden에서 출력으로 가능 가중치랑 bias를 정해준다.
o_w31 = tf.truncated_normal(shape=(3,1), mean=0, stddev=1.0, dtype=tf.float32) # o_w31: output으로 가는데 3개가 1개로 가는 것
o_b11 = tf.zeros(shape=(1,1)) # o_b11 : bias 1, 출력 1
# 변수만듬
w2 = tf.Variable(o_w31)
b2 = tf.Variable(o_b11)

##### weight와 bias는 결정되고 이제 신경망 네트워크를 설계한다.
z1 = tf.matmul(x, w1) + b1 # 첫번째 hidden Node에서의 출력은 z1이라한다.
a1 = tf.sigmoid(z1) # 그냥 쓰는게 아니고 sigmoid 함수를 통과시킨다.

z2 = tf.matmul(a1, w2) + b2 # hidden Layer에서 outout layer를 계산한다.
a2 = tf.sigmoid(z2)

# 최종적인 결과물을 y-hat이라고 저장한다.
yhat = a2

##### 네트워크를 설계햇고 Loss 를 정의한다. - crossEntropy로 정의한다.
loss = -y * tf.log(yhat) - (1-y) * tf.log(1-yhat) # y는 4x1 yhat도 4x1
loss = tf.reduce_mean(loss) # 이것들의 평균을 낸다.

##### 이 loss에 대해서 최적화를 한다.
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)    # 경사하강법을 쓴다. (파라미터로 learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)

##### 이제 이 optimizer를 최소화하는 것을 반복해준다.
train = optimizer.minimize(loss)




##### 여기까지 학습준비를 했고, 학습이 될때 정확도가 얼마가 되는지 측정한다.
# 출력노드가 1개고 나오는 값이 sigmoid 함수를 통과하기 때문에 0~1 사이의 값이된다. 
# 기준값으로 0.5부터 크면 1번클래스, 작으면 2번클래스로 정의하기때문에
# 큰지 작은지 결정하는 thresholding을 정의해준다.
thresholding = yhat > 0.5 # yhat이 0.5보다 크냐 (thresholding = tf.greater_equl(yhat, 0.5)   이렇게 해줘도됨)
# 4x1 이기때문에 T/F가 나와았다. T:1.0, F:0.0으로 바꾸자.
thresholding = tf.cast(thresholding, tf.float32 )  # cast 함수 : thresholding 값을 tf.float32로 바꿔줌
# 결과물이 traning data target value(y)와 똑같은지 확인한다.
prediction = tf.cast(tf.equal(thresholding, y), tf.float32)
# Accuracy는 이것들의 평균을 구하면된다.
accuracy = tf.reduce_mean(prediction)


##### 이제 학습할 일만 남았다.
# Session 만들어주고
sess = tf.Session()
# 제일먼저 변수초기화해준다.
sess.run(tf.global_variables_initializer())
 # 반복을 위해 EPOCH을 300준다.

Loss = []
EPOCH = 200
for i in range(EPOCH):
    outs = sess.run([train, loss, yhat, accuracy], feed_dict={x: x_data, y: y_data})   # ([계산을 해야할 것 들], x와 y의 data를 준다.)
    print(f'i : {i}, loss : {outs[1]}, accuracy : {outs[3]}')
    Loss.append(outs[1])


plt.plot(Loss[:])
plt.show()

sess.close()
    

# 결과
'''
1. 
Hyper Parameter
 - np.random.seed(123)
 - tf.set_random_seed(123)
 - learning_rate = 0.01
 
결과
 - 뒤로갈수록 정확도가 떨어졌다.
Loss 
 - 떨어짐 : Epoch를 늘린다. => 1000번정도

2. 
Epoch 1000
결과
 - 수렴 0.5 ㅠㅠ
3.
Epoch 50000
결과
 - accuracy : 1.0 !!!
 - Loss는 0.05대로 떨어짐

결론. Loss그래프를 보고 학습을 더해야할지 말아야할지 판단할 수 있다
    iter를 많이 했음에도 불구하고 loss가 안떨어진다면 learning_rate를 좀더 작게할 필요가 있다.
    하나더, 50000번을 돌렸더니 loss가 굉장히 떨어졌지만 이전의 EPOCH에 대해서는 잘 나타낼지는 미지수다.
'''