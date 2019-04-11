import sys

import keras

from chip2018.chip_lstm import ChipLSTM
from chip2018.chip_gru import ChipGRU
from chip2018.chip_bilstm import ChipBiLSTM

sys.path.append('..')
from utils.word2vec_fast import *

# 加载词向量
wv = Word2VecFast.load_word2vec_format(file_path='../data/chip2018/word_embedding.txt', word_shape=300)
print('word_embedding shape: ', wv.word_shape())
word_size, embedding_size = wv.word_embeddings().shape[0], wv.word_embeddings().shape[1]

# 加载数据集
dataset = joblib.load(filename='../data/chip2018/chip2018.data2')
X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']
print('X_train shape:', X_train.shape)
Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)

a = ChipLSTM(45, 2, wv)
a.fit(X_train, Y_train, X_test, Y_test)
