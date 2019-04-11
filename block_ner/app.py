import sys

import tensorflow as tf
from tensorflow.contrib.crf import crf_decode
from tensorflow.contrib.crf import crf_log_likelihood

sys.path.append('..')
from utils.word2vec_fast import *

hidden_size_lstm = 128
tags = 10
epochs = 10

wv = Word2VecFast.load_word2vec_format(file_path='../data/chip2018/word_embedding.txt', word_shape=300)
print('word_embedding shape: ', wv.word_shape())
word_size, embedding_size = wv.word_embeddings().shape[0], wv.word_embeddings().shape[1]

# 网络结构
input_x = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
input_x_seq_len = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
dropout = tf.placeholder(tf.float32, name="dropout")
input_y = tf.placeholder(tf.int32, shape=[None, None], name="labels")

with tf.name_scope("embedding"):
    WL_word2vec = tf.constant(wv.word_embeddings(), dtype=tf.float32)
    WL_word_embedding = tf.nn.embedding_lookup(WL_word2vec, input_x)

with tf.variable_scope("bi_lstm"):
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_lstm)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_lstm)
    (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                        WL_word_embedding,
                                                                        dtype=tf.float32,
                                                                        sequence_length=input_x_seq_len)
    output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
    output = tf.nn.dropout(output, dropout)

with tf.variable_scope("proj"):
    W = tf.get_variable("W", dtype=tf.float32, shape=[2 * hidden_size_lstm, tags])
    b = tf.get_variable("b", shape=[tags], dtype=tf.float32, initializer=tf.zeros_initializer())

    nsteps = tf.shape(output)[1]
    output = tf.reshape(output, [-1, 2 * hidden_size_lstm])
    pred = tf.matmul(output, W) + b
    logits = tf.reshape(pred, [-1, nsteps, tags])

with tf.name_scope("crf"):
    log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
                                                           tag_indices=input_y,
                                                           sequence_lengths=input_x_seq_len)
    transition_params = tf.add(transition_params, 0)
    loss_op = tf.reduce_mean(-log_likelihood, name="loss")

with tf.name_scope("output"):
    output, _ = crf_decode(logits, transition_params=transition_params, sequence_length=input_x_seq_len)
    output = tf.add(output, 0, name="output")

with tf.variable_scope("train"):
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)


def pad_sequences(sequences, pad_tok, nlevels=1):
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def get_feed_dict(self, words, labels=None):
    word_ids, sequence_lengths = pad_sequences(words, 0)

    # build feed dictionary
    feed = {
        input_x: None,
        input_x_seq_len: None,
        dropout: 0.1
    }
    if labels is not None:
        labels, _ = pad_sequences(labels, 0)
        feed[self.labels] = labels

    return feed, sequence_lengths


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, epochs))
        score = run_epoch(train, dev, epoch)
