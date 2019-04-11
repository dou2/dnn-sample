import keras
import tensorflow as tf
from keras import datasets

from utils.all import *

print(tf.VERSION)


class MultiClassify(object):
    def __init__(self, n_dimension, n_class):
        self.n_dimension = n_dimension
        self.n_class = n_class
        self.__init_var()

    def __init_var(self):
        self._x = tf.placeholder(tf.float32, shape=(None, self.n_dimension), name='input')
        self._y = tf.placeholder(tf.float32, shape=(None, self.n_class))

        self.L1 = tf.layers.Dense(512, activation=tf.nn.relu)(self._x)
        self.L2 = tf.layers.Dense(256, activation=tf.nn.relu)(self.L1)
        self.L3 = tf.layers.Dense(self.n_class)(self.L2)
        self.logits = tf.nn.softmax(self.L3, name='logits')
        self.output = tf.argmax(self.logits, axis=1, name='output')

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self.L3))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_op)
        self.precision_op = tf.reduce_mean(tf.cast(tf.equal(self.output, tf.argmax(self._y, axis=1)), tf.float32))

        self.a1 = tf.argmax(self._y, axis=1)
        self.c1 = self.a1 * self.output
        # 准确率，召回率
        self.acc_op = tf.count_nonzero(self.c1) / tf.count_nonzero(self.output)
        self.recall_op = tf.count_nonzero(self.c1) / tf.count_nonzero(self.a1)
        self.init_op = tf.global_variables_initializer()

    def fit(self, x=None, y=None, epochs=1, validation_data=None, export_dir=None):
        with tf.Session() as session:
            session.run(self.init_op)

            for epoch in range(epochs):
                print('Epoch ', (epoch + 1), '/', epochs)
                session.run(self.train_op, feed_dict={self._x: x, self._y: y})
                loss, acc, recall = session.run([self.loss_op, self.acc_op, self.recall_op],
                                                feed_dict={self._x: x, self._y: y})
                if validation_data and len(validation_data) == 2:
                    val_loss, val_acc, val_recall = session.run([self.loss_op, self.acc_op, self.recall_op],
                                                                feed_dict={self._x: validation_data[0],
                                                                           self._y: validation_data[1]})
                    print('loss: ', "{:.4f}".format(loss), ' - acc: ', "{:.4f}".format(acc), ' - recall: ',
                          "{:.4f}".format(recall), ' - val_loss: ',
                          "{:.4f}".format(val_loss), ' - val_acc: ', "{:.4f}".format(val_acc), ' - val_recall: ',
                          "{:.4f}".format(val_recall))
                else:
                    print('loss: ', "{:.4f}".format(loss), ' - acc: ', "{:.4f}".format(acc))
            if export_dir:
                save_model(session, export_dir)


if __name__ == '__main__':
    # 加载数据集
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    # min-max归一化
    X_test = X_test / 255
    X_test = X_test.reshape(10000, 784)
    y_test = keras.utils.to_categorical(y_test, 10)

    # num_examples = X_train.shape[0]
    # total_batch = int(num_examples / batch_size)
    a = MultiClassify(784, 10)
    a.fit(X_test, y_test, epochs=100)
