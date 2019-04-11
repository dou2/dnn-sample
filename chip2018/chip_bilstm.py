import tensorflow as tf

from chip2018.chip_rnn import ChipRNN


class ChipBiLSTM(ChipRNN):
    def encoder(self):
        lstm_fw_cell = self.get_rnn_cell()
        lstm_bw_cell = self.get_rnn_cell()
        with tf.variable_scope("lstm"):
            (output1, states1) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                 self.WL_word_embedding1,
                                                                 dtype=tf.float32,
                                                                 sequence_length=self.input_x1_seq_len)
            output1 = tf.concat([states1[0].h, states1[1].h], axis=-1)
        with tf.variable_scope("lstm", reuse=True):
            (output2, states2) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                 self.WL_word_embedding2,
                                                                 dtype=tf.float32,
                                                                 sequence_length=self.input_x2_seq_len)
            output2 = tf.concat([states2[0].h, states2[1].h], axis=-1)
        return tf.concat([tf.abs(output1 - output2), ], axis=-1)

    def get_rnn_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
