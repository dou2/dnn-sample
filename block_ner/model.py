import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood

from utils.word2vec_fast import *


# from keras.preprocessing.sequence import pad_sequences


def pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]
    return sequence_padded, sequence_length


class BiLSTM_CRF(object):
    def __init__(self):
        self.__init_tags()
        self.hidden_dim = 100
        self.tag_num = len(self.idx_to_tag)
        self.vec_file = 'E:\sequence_tagging-master\data\glove.6B\glove.6B.50d.txt'
        self.wv = Word2VecFast.load_word2vec_format(file_path=self.vec_file, word_shape=50)

    def build(self):
        self.__init_placeholders()
        self.__init_lookup()
        self.__init_bi_lstm()
        self.__init_loss_op()
        self.__init_train_op()
        self.__init_session()

    def __init_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.sequence_len = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_len")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")

    def __init_lookup(self):
        with tf.name_scope("embedding"):
            self.WL_word2vec = tf.constant(self.wv.word_embeddings(), dtype=tf.float32)
            self.word_embeddings = tf.nn.embedding_lookup(self.WL_word2vec, self.word_ids)

    def __init_bi_lstm(self):
        with tf.variable_scope("bi_lstm"):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                                self.word_embeddings,
                                                                                dtype=tf.float32,
                                                                                sequence_length=self.sequence_len)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            w = tf.get_variable(name="w", shape=[2 * self.hidden_dim, self.tag_num], dtype=tf.float32)
            b = tf.get_variable(name="b", shape=[1, self.tag_num], dtype=tf.float32)

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, w) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.tag_num])

    def __init_loss_op(self):
        with tf.name_scope("crf"):
            log_likelihood, transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_len)
            self.transition_params = tf.add(transition_params, 0)
            self.loss_op = tf.reduce_mean(-log_likelihood, name="loss")

    def __init_train_op(self):
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss_op)

    def train(self):
        seqs, tags = self.load_seq()
        for epoch in range(100):
            fd, _ = self.get_feed_dict(seqs, tags)
            _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict=fd)
            print("train loss: ", loss)

    def __init_tags(self):
        with open(file='tags.txt', encoding='utf-8') as f:
            self.tag_to_idx = {tag.strip(): idx for idx, tag in enumerate(f)}
            self.idx_to_tag = {idx: tag for idx, tag in enumerate(self.tag_to_idx)}

    def __init_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def load_seq(self):
        sequence_list = []
        tag_list = []
        with open(file='test.txt', encoding='utf-8') as f:
            words = []
            tags = []
            for (line_num, line) in enumerate(f):
                ss = line.strip().split()
                if len(ss) == 0:
                    sequence_list.append(words)
                    tag_list.append(tags)
                    words = []
                    tags = []
                else:
                    word, tag = ss[0], ss[1]
                    wid = self.wv.words_id(word.lower())[0]
                    tid = self.tag_to_idx[tag]
                    words.append(wid)
                    tags.append(tid)
        return sequence_list, tag_list

    def get_feed_dict(self, sequences, tags=None):
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = pad_sequences(sequences=sequences, pad_tok=0,
                                                         max_length=max_length)
        feed = {
            self.word_ids: sequence_padded,
            self.sequence_len: sequence_length,
            self.dropout: 0.1
        }
        if tags is not None:
            feed[self.labels], _ = pad_sequences(sequences=tags, pad_tok=0,
                                                 max_length=max_length)
        return feed, sequence_length

    def processing_word(self, word):
        return self.wv.words_id(word.lower())[0]

    def predict(self, words_raw):
        words = [self.processing_word(w) for w in words_raw]
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]
        return preds

    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words)
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
            [self.logits, self.transition_params], feed_dict=fd)

        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths


if __name__ == '__main__':
    a = BiLSTM_CRF()
    a.build()
    a.train()
    b = a.predict(words_raw=['Jean', 'lives', 'in', 'New', 'York'])
    print(b)
