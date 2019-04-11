import sys

import keras
import tensorflow as tf

sys.path.append('..')
from utils.word2vec_fast import *

sequence_length = 45
num_classes = 2
learning_rate = 0.005
num_epoch = 5
batch_size = 500

# 加载数据集
dataset = joblib.load(filename='../data/chip2018/chip2018.data2')
X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
print('X_train shape:', X_train.shape)
Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)

# 加载词向量
wv = Word2VecFast.load_word2vec_format(file_path='../data/chip2018/word_embedding.txt', word_shape=300)
print('word_embedding shape: ', wv.word_shape())
word_size, embedding_size = wv.word_embeddings().shape[0], wv.word_embeddings().shape[1]

# 定义网络结构
input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
input_x1_sequences = tf.placeholder(dtype=tf.int32, shape=[None], name="inputs_x1_sequences")
input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
input_x2_sequences = tf.placeholder(dtype=tf.int32, shape=[None], name="inputs_x2_sequences")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

# 嵌入层
WL_word2vec = tf.constant(wv.word_embeddings(), dtype=tf.float32)
WL_word_embedding1 = tf.nn.embedding_lookup(WL_word2vec, input_x1)
WL_word_embedding2 = tf.nn.embedding_lookup(WL_word2vec, input_x2)

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)

with tf.variable_scope("a"):
    (output1, state1) = tf.nn.dynamic_rnn(rnn_cell, inputs=WL_word_embedding1, dtype=tf.float32,
                                          sequence_length=input_x1_sequences)
with tf.variable_scope("a", reuse=True):
    (output2, state2) = tf.nn.dynamic_rnn(rnn_cell, inputs=WL_word_embedding2, dtype=tf.float32,
                                          sequence_length=input_x2_sequences)

concreted = tf.abs(state1.h - state2.h)

logits = tf.layers.dense(inputs=concreted, units=2)
# logits = tf.clip_by_value(logits, 1)
y_predict = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=input_y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

correct_predict = tf.equal(tf.argmax(y_predict, axis=1), tf.argmax(input_y, axis=1))
acc_op = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

init_op = tf.global_variables_initializer()

total_batch = int(X_train.shape[0] / batch_size)


def next_batch(i):
    a = i * batch_size
    b = (i + 1) * batch_size
    return X_train[a:b], Y_train[a:b]


with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(num_epoch):
        avg_cost = 0.
        avg_acc = 0.
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(i)
            b_input_x1 = batch_xs[:, 0:45]
            b_input_x2 = batch_xs[:, 45:]
            len1 = np.count_nonzero(np.where(b_input_x1 == 9646, 0, 1), axis=1)
            len2 = np.count_nonzero(np.where(b_input_x2 == 9646, 0, 1), axis=1)
            sess.run(train_op, feed_dict={input_x1: b_input_x1, input_x2: b_input_x2, input_y: batch_ys,
                                          input_x1_sequences: len1, input_x2_sequences: len2})
            c, acc = sess.run([loss_op, acc_op],
                              feed_dict={input_x1: b_input_x1, input_x2: b_input_x2, input_y: batch_ys,
                                         input_x1_sequences: len1, input_x2_sequences: len2})
            avg_cost += c / total_batch
            avg_acc += acc / total_batch

            b_input_test_x1 = X_test[:, 0:45]
            b_input_test_x2 = X_test[:, 45:]
            len1_test = np.count_nonzero(np.where(b_input_test_x1 == 9646, 0, 1), axis=1)
            len2_test = np.count_nonzero(np.where(b_input_test_x2 == 9646, 0, 1), axis=1)
            c1, acc1 = sess.run([loss_op, acc_op],
                                feed_dict={input_x1: b_input_test_x1, input_x2: b_input_test_x2, input_y: Y_test,
                                           input_x1_sequences: len1_test, input_x2_sequences: len2_test})
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c), "accuracy=", "{:.9f}".format(acc),
                  "test_loss=", "{:.9f}".format(c1), "test_accuracy=", "{:.9f}".format(acc1))
    print("Optimization Finished!")
