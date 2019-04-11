import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

from utils import all

cancer_data = datasets.load_breast_cancer()
X, Y = cancer_data.data, cancer_data.target
Y = np.reshape(Y, [Y.shape[0], 1])
x_train, x_test, y_train, y_test = all.train_test_split(X, Y)
m, n = x_train.shape

_x = tf.placeholder(tf.float32, shape=(None, n))
_y = tf.placeholder(tf.float32, shape=(None, 1))

L1 = tf.layers.Dense(1)(_x)
logits = tf.nn.sigmoid(L1)
y_predict = tf.where(logits < 0.5, tf.zeros_like(logits), tf.ones_like(logits))

loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=_y, logits=L1))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)
acc_op = 1 - tf.reduce_mean(tf.abs(y_predict - _y))
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    for i in range(10000):
        session.run(train_op, feed_dict={_x: x_train, _y: y_train})
        loss, acc = session.run([loss_op, acc_op], feed_dict={_x: x_train, _y: y_train})
        loss_test, acc_test = session.run([loss_op, acc_op], feed_dict={_x: x_test, _y: y_test})
        if i % 100 == 0:
            print('epoch: ', '%04d' % (i + 1), 'loss: ', "{:.4f}".format(loss), 'acc: ', "{:.4f}".format(acc),
                  'loss_test: ', "{:.4f}".format(loss_test), 'acc_test: ', "{:.4f}".format(acc_test))
    # 输出预测
    test_correct = session.run(y_predict, feed_dict={_x: x_test, _y: y_test})
    plt.plot(test_correct)
    plt.plot(y_test)
    plt.show()
