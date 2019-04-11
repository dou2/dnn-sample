import sys

import keras
from keras.layers import Embedding, LSTM, Dense, Subtract
from keras.models import Model, Sequential

sys.path.append('..')
from utils.word2vec_fast import *

sequence_length = 45
num_classes = 2

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
wv_initializer = keras.initializers.Constant(wv.word_embeddings())

encoder_a = Sequential()
encoder_a.add(Embedding(input_dim=word_size, output_dim=embedding_size, embeddings_initializer=wv_initializer))
encoder_a.add(LSTM(128))

encoder_b = Sequential()
encoder_a.add(Embedding(input_dim=word_size, output_dim=embedding_size, embeddings_initializer=wv_initializer))
encoder_a.add(LSTM(128))

decoder = Sequential()
decoder.add(Subtract()([encoder_a, encoder_b]))
decoder.add(Dense(num_classes, activation='softmax'))

decoder.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['acc'])
history = decoder.fit(x=[X_train[:, 0:45], X_train[:, 45:]], y=Y_train, batch_size=1000, epochs=5,
                      validation_data=([X_test[:, 0:45], X_test[:, 45:]], Y_test))
