import tensorflow as tf

from chip2018.chip_rnn import ChipRNN


class ChipLSTM(ChipRNN):
    def encoder(self):
        rnn_cell = self.get_rnn_cell()
        with tf.variable_scope("lstm"):
            (output1, state1) = tf.nn.dynamic_rnn(rnn_cell, self.WL_word_embedding1, dtype=tf.float32,
                                                  sequence_length=self.input_x1_seq_len)
        with tf.variable_scope("lstm", reuse=True):
            (output2, state2) = tf.nn.dynamic_rnn(rnn_cell, self.WL_word_embedding2, dtype=tf.float32,
                                                  sequence_length=self.input_x2_seq_len)
        return tf.abs(state1.h - state2.h)
        # return tf.nn.l2_normalize(state1.h) * tf.nn.l2_normalize(state2.h)

    def get_rnn_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
