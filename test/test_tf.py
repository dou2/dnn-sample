import numpy as np
import tensorflow as tf
from ..utils.word2vec_fast import *


def test_cross_entropy():
    y = np.array([0, 0, 1, 0])
    y_predict = np.array([3.5, 2.1, 7.89, 4.4])

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict)
    loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_predict)
    loss3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y), logits=y_predict)

    with tf.Session() as sess:
        print(sess.run(loss))
        print(sess.run(loss2))
        print(sess.run(loss3))
        print(sess.run(tf.argmax(y)))
        print(sess.run(tf.nn.softmax(y_predict, axis=0)))


wv = Word2VecFast.load_word2vec_format(file_path='../data/chip2018/word_embedding.txt')
print(wv.words_id('W103948 W107108'))
