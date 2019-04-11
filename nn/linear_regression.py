import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
m, n = x_train.shape
y_train = np.reshape(y_train, [y_train.shape[0], 1])
y_test = np.reshape(y_test, [y_test.shape[0], 1])

x_ph = tf.placeholder(tf.float32, shape=(None, n))
y_ph = tf.placeholder(tf.float32, shape=(None, 1))

y_predict = tf.layers.dense(units=1, inputs=x_ph)

# y_predict = tf.nn.bias_add(tf.matmul(x_ph, w), b)
loss_op = tf.reduce_mean(tf.square(y_ph - y_predict))
train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss_op)
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    for i in range(20000):
        _, loss = session.run([train_op, loss_op], feed_dict={x_ph: x_train, y_ph: y_train})
        print(loss)
    # 输出预测
    test_predict = session.run(y_predict, feed_dict={x_ph: x_test, y_ph: y_test})
    plt.plot(test_predict)
    plt.plot(y_test)
    plt.show()
