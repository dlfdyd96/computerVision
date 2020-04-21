import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

# 파일에서 데이터 읽어옴
data = np.loadtxt('xor_data.txt')
x_data = data[:, :-1]
y_raw_data= data[:, -1].astype(np.int32)
y_data = np.zeros(shape=(y_raw_data.shape[0], 2))
y_data[np.arange(y_raw_data.shape[0]), y_raw_data] = 1


# 학습 데이터 생성
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
learning_rate = 0.01


# 테스트 데이터
pt = np.arange(0, 1, 0.01)
pt_x, pt_y = np.meshgrid(pt, pt) # 정방행렬
pt_shape = pt_x.shape
pt_x = pt_x.flatten()
pt_y = pt_y.flatten()

x_test = np.vstack((pt_x, pt_y)).T
y_test = np.zeros(shape=(x_test.shape[0], 2))

x = tf.placeholder(tf.float32, shape=[None, 2]) 
y = tf.placeholder(tf.float32, shape=[None, 2])

fw = tf.keras.initializers.truncated_normal()
fb = tf.keras.initializers.Zeros()

w1 = tf.Variable(fw(shape=(2,15)), dtype=tf.float32)
b1 = tf.Variable(fb(shape=(1,15)), dtype=tf.float32)
w2 = tf.Variable(fw(shape=(15,2)), dtype=tf.float32)
b2 = tf.Variable(fb(shape=(1,2)), dtype=tf.float32)

z1 = tf.matmul(x, w1) + b1 
a1 = tf.sigmoid(z1) 

z2 = tf.matmul(a1, w2) + b2 
a2 = tf.sigmoid(z2)

yhat = a2 # 최종 결과물

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z2)
loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(yhat, axis=1))
correct = tf.cast(correct, tf.float32)
accuracy = tf.reduce_mean(correct)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)


# 학습
EPOCH = 1000
for i in range(EPOCH):
    outs = sess.run([train, loss, yhat, accuracy], feed_dict={x: x_data, y: y_data})
    if(i % 100 == 0):
        print(f'i : {i}, loss : {outs[1]}, accuracy : {outs[3]}')
    

# 테스트
cost, y_ = sess.run([loss, yhat], feed_dict={x: x_test, y: y_test})
labels = np.argmax(y_, axis=1).astype(np.int32)


# 테스트 결과 출력
colors = ['r', 'b']
# labels = np.array([colors[i] for i in labels])
'''
plt.scatter(pt_x, pt_y, c= labels)
plt.show()
'''
pt_x = pt_x.reshape(pt_shape)
pt_y = pt_y.reshape(pt_shape)
labels = labels.reshape(pt_shape)
plt.contourf(pt_x, pt_y, labels, cmap=plt.cm.Spectral)
plt.show()

sess.close()
