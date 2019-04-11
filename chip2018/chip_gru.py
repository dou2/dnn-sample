import tensorflow as tf

from chip2018.chip_rnn import ChipRNN


class ChipGRU(ChipRNN):
    def get_rnn_cell(self):
        return tf.nn.rnn_cell.GRUCell(self.cell_size)
