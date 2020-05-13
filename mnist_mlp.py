import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# 원핫벡터변환
def one_hot(labels, targets=10):
    samples=len(labels)
    out=np.zeros((samples,targets))
    out[range(samples), labels] = 1
    return out

# 데이터 로딩
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
train = mnist[0]
test = mnist[1]
x_train = train[0].astype(np.float32) / 255.0
y_train = train[1]
x_test = test[0].astype(np.float32) / 255.0
y_test = test[1]

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

# 하이퍼 파라미터
learning_rate = 0.05
NODES = 100

# 네트워크 설계
x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input') # [None, 784] : sample 몇개인지 몰라도 알아서 다됨 & 입력은 784
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='target')

w1 = tf.Variable(tf.truncated_normal(shape=[784, NODES], stddev=0.1), name='weight')
b1 = tf.Variable(tf.zeros(shape=[NODES], dtype=tf.float32), name='bias')

w2 = tf.Variable(tf.truncated_normal(shape=[NODES, 10], stddev=0.1), name='weight2')
b2 = tf.Variable(tf.zeros(shape=[10]), name='bias2')

z1 = tf.matmul(x, w1) + b1
a1 = tf.sigmoid(z1) # 학습이 안될 수 도 있다.

z2 = tf.matmul(a1, w2) + b2
yhat = tf.nn.softmax(z2)

# 손실함수, 최적화방법 결정
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z2)
loss = tf.reduce_mean(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# 인식성능계산
correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 세션 생성
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 에포크 / 스텝은 다르다.
# 그룹(batch)을 지어서 학습 하는 것이 추세. (local minimum에 빠질 위험이 적음)
# ex 6만개중에 10개를 확률 적으로 뽑아서 GradentDescent 적용 -> Stocastic Gradient Descent Method 라고 함.
# local Minimum에 빠지는 것을 막아주는 효과가 있다.
EPOCH = 50
BATCH = 100
train_samples = x_train.shape[0]
steps = train_samples // BATCH # =60번

x_train_batch = x_train.reshape(-1, BATCH, 28*28)
y_train_batch = y_train.reshape(-1, BATCH, 10)
# x_train_batch: (60, 1000, 784)
# y_train_batch: (60, 1000, 10)

x_test = x_test.reshape(-1, 28*28)
y_test = y_test.reshape(-1, 10)

loss_epoch = []
accuracy_epoch = []

loss_test = []
accuracy_test = []



for epoch in range(1, EPOCH + 1):
    # 학습
    loss_batch = []
    accuracy_batch = []

    for step in range(steps):
        x_batch = x_train_batch[step]
        y_batch = y_train_batch[step]
        _, loss_, acc_ = sess.run([train, loss, accuracy], feed_dict={x: x_batch, y:y_batch})
        print(f'epoch={epoch}, step={step}, loss={loss_}, accuracy={acc_}')

        loss_batch.append(loss_)
        accuracy_batch.append(acc_)


    # 모든 스텝을 끝나면 한개의 에폭이 됨    
    mean_loss = np.mean(loss_batch)
    mean_accuracy = np.mean(accuracy_batch)
    loss_epoch.append(mean_loss)
    accuracy_epoch.append(mean_accuracy)

    print(f'train: loss={mean_loss:.8f} accuracy={mean_accuracy:.5f}')

    # 테스트 (test)
    loss_, acc_ = sess.run([loss, accuracy], feed_dict={x: x_batch, y:y_batch})
    loss_test.append(loss_)
    accuracy_test.append(acc_)
    print(f'test: loss={loss_:.8f} accuracy={acc_:.5f}')

sess.close()


plt.plot(loss_epoch, 'r')
plt.plot(loss_test, 'b')
plt.show()

plt.plot(accuracy_epoch, 'r')
plt.plot(accuracy_test, 'b')

plt.show()