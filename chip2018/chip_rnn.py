import numpy as np
import tensorflow as tf


class ChipRNN(object):
    def __init__(self, sequence_length, num_classes, wv):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.wv = wv
        self.cell_size = 128
        self.learning_rate = 0.005
        self.padding = self.wv.words_id('<UNK>')
        self.__init_var()
        self.__build_model()

    def __init_var(self):
        self.input_x1 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")
        self.input_x1_seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="input_x1_seq_len")
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")
        self.input_x2_seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="input_x2_seq_len")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32)
        # 嵌入层
        self.WL_word2vec = tf.constant(self.wv.word_embeddings(), dtype=tf.float32)
        self.WL_word_embedding1 = tf.nn.embedding_lookup(self.WL_word2vec, self.input_x1)
        self.WL_word_embedding2 = tf.nn.embedding_lookup(self.WL_word2vec, self.input_x2)
        # Attention
        self.attention = tf.Variable(tf.random_normal([self.cell_size*2], stddev=1), dtype=tf.float32)

    def __build_model(self):
        merged = self.encoder()
        with_attention = merged * self.attention
        # h_drop = tf.nn.dropout(merged_layer, self.keep_prob)
        output = tf.layers.dense(inputs=with_attention, units=2)
        y_predict = tf.nn.softmax(output)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=self.input_y))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_op)
        self.correct_predict = tf.equal(tf.argmax(y_predict, axis=1), tf.argmax(self.input_y, axis=1))
        self.acc_op = tf.reduce_mean(tf.cast(self.correct_predict, tf.float32))
        self.init_op = tf.global_variables_initializer()

    def fit(self, x_train, y_train, x_test, y_test, num_epoch=5, batch_size=500):
        total_batch = int(x_train.shape[0] / batch_size)

        with tf.Session() as sess:
            sess.run(self.init_op)

            for epoch in range(num_epoch):
                for i in range(total_batch):
                    a = i * batch_size
                    b = (i + 1) * batch_size
                    batch_xs, batch_ys = x_train[a:b], y_train[a:b]
                    b_input_x1 = batch_xs[:, 0:self.sequence_length]
                    b_input_x2 = batch_xs[:, self.sequence_length:]
                    len1 = np.count_nonzero(np.where(b_input_x1 == self.padding, 0, 1), axis=1)
                    len2 = np.count_nonzero(np.where(b_input_x2 == self.padding, 0, 1), axis=1)
                    sess.run(self.train_op,
                             feed_dict={self.input_x1: b_input_x1, self.input_x2: b_input_x2, self.input_y: batch_ys,
                                        self.keep_prob: 0.8,
                                        self.input_x1_seq_len: len1, self.input_x2_seq_len: len2})
                    c, acc = sess.run([self.loss_op, self.acc_op],
                                      feed_dict={self.input_x1: b_input_x1, self.input_x2: b_input_x2,
                                                 self.input_y: batch_ys, self.keep_prob: 0.8,
                                                 self.input_x1_seq_len: len1, self.input_x2_seq_len: len2})
                    b_input_test_x1 = x_test[:, 0:self.sequence_length]
                    b_input_test_x2 = x_test[:, self.sequence_length:]
                    len1_test = np.count_nonzero(np.where(b_input_test_x1 == self.padding, 0, 1), axis=1)
                    len2_test = np.count_nonzero(np.where(b_input_test_x2 == self.padding, 0, 1), axis=1)
                    c1, acc1 = sess.run([self.loss_op, self.acc_op],
                                        feed_dict={self.input_x1: b_input_test_x1, self.input_x2: b_input_test_x2,
                                                   self.input_y: y_test, self.keep_prob: 0.8,
                                                   self.input_x1_seq_len: len1_test, self.input_x2_seq_len: len2_test})
                    print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c), "accuracy=",
                          "{:.9f}".format(acc),
                          "test_loss=", "{:.9f}".format(c1), "test_accuracy=", "{:.9f}".format(acc1))

    def encoder(self):
        rnn_cell = self.get_rnn_cell()
        (output1, state1) = tf.nn.dynamic_rnn(rnn_cell, self.WL_word_embedding1, dtype=tf.float32,
                                              sequence_length=self.input_x1_seq_len)
        (output2, state2) = tf.nn.dynamic_rnn(rnn_cell, self.WL_word_embedding2, dtype=tf.float32,
                                              sequence_length=self.input_x2_seq_len)
        return tf.abs(state1 - state2)

    def get_rnn_cell(self):
        return tf.nn.rnn_cell.BasicRNNCell(self.cell_size)
