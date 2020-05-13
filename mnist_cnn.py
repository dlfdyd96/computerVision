import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# 원핫벡터변환
def one_hot(labels, targets=10):
    samples = len(labels)
    out = np.zeros((samples, targets))
    out[range(samples), labels] = 1
    return out


# 데이터 로딩
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
train = mnist[0]
test = mnist[1]

x_train = train[0].astype(np.float32) / 255
y_train = train[1]
x_test = test[0].astype(np.float32) / 255
y_test = test[1]

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

INPUT = 784
OUTPUT = 10

#
# 하이퍼파라미터
#
learning_rate = 0.05
#K1 = 32
#K2 = 64
#F1 = 100
# KERNEL_SIZE=5

#
# 학습 가중치
#
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# 이미지 변환 식
x_image = tf.reshape(x, [-1, 28, 28, 1])


''' 학습 가중치 '''
# 첫번째 컨벌루션 레이어
w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# 두번째 컨벌루션 레이어
w_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

# 첫번째 완전연결 레이어
w_full1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 100], stddev=0.1))
b_full1 = tf.Variable(tf.zeros(shape=[100], dtype=tf.float32))

# 두번째 완전연결 레이어
w_full2 = tf.Variable(tf.truncated_normal(shape=[100, 10], stddev=0.1))
b_full2 = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32))

''' 네트워크 구조 '''
# 첫번째 컨벌루션 레이어
conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 두번째 컨벌루션 레이어
conv2 = tf.nn.conv2d(pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

# 첫번째 완전연결 레이어
z1 = tf.matmul(pool2_flat, w_full1) + b_full1
a1 = tf.nn.relu(z1)

# 두번째 완전연결 레이어
z2 = tf.matmul(a1, w_full2) + b_full2
yhat = tf.nn.softmax(z2)

''' 손실함수 및 최적화 '''
# 손실함수 및 최적화
#
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z2)
loss = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# 인식률
#
correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

''' 학습 및 테스트 '''
#
# 세션 생성
#
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 학습
#
# 에포크 / 스텝
#
EPOCH = 2
BATCH = 100
train_samples = x_train.shape[0]
steps = train_samples // BATCH

x_train_batch = x_train.reshape(-1, BATCH, 28, 28, 1)
y_train_batch = y_train.reshape(-1, BATCH, 10)

x_test = x_test.reshape(-1, 28, 28, 1)
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
        _, loss_, acc_ = sess.run([train, loss, accuracy], feed_dict={x: x_batch, y: y_batch})
        print(f'epoch={epoch:4d}, step={step:4d}, loss={loss_:12.8f}, accuracy={acc_:6.5f}')

        loss_batch.append(loss_)
        accuracy_batch.append(acc_)
    # 모든 스텝을 끝나면 한개의 에폭이 됨
    mean_loss = np.mean(loss_batch)
    mean_accuracy = np.mean(accuracy_batch)
    loss_epoch.append(mean_loss)
    accuracy_epoch.append(mean_accuracy)

    print(f'train: loss={mean_loss:.8f} accuracy={mean_accuracy:.5f}')

    # 테스트 (test)
    loss_, acc_ = sess.run([loss, accuracy], feed_dict={x: x_batch, y: y_batch})
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
