import sys

import keras
from keras.layers import Input, Embedding, LSTM, Dense, Multiply
from keras.models import Model

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

shared_lstm = LSTM(128, activation='tanh')

input_x1 = Input(shape=(sequence_length,), dtype='int32', name='input_x1')
embedding_x1 = Embedding(input_dim=word_size, output_dim=embedding_size, embeddings_initializer=wv_initializer,
                         trainable=False)(
    input_x1)
lstm_out1 = shared_lstm(embedding_x1)

input_x2 = Input(shape=(sequence_length,), dtype='int32', name='input_x2')
embedding_x2 = Embedding(input_dim=word_size, output_dim=embedding_size, embeddings_initializer=wv_initializer,
                         trainable=False)(
    input_x2)
lstm_out2 = shared_lstm(embedding_x2)


def layers_abs(x1):
    return keras.backend.abs(x1[0] - x1[1])


def layers_mul(x1):
    return keras.backend.l2_normalize(x1[0]) * keras.backend.l2_normalize(x1[1])


merged = keras.layers.Lambda(layers_mul)([lstm_out1, lstm_out2])
# merged = Multiply()([keras.utils.normalize(lstm_out1) * keras.utils.normalize(lstm_out2)])
# merged = Multiply()([lstm_out1, lstm_out2])
print(merged)
output = Dense(2, activation='softmax')(merged)
print(output)

model = Model(inputs=[input_x1, input_x2], outputs=output)
model.summary()
# 批量梯度下降
model.compile(optimizer=keras.optimizers.Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['acc'])
keras.utils.plot_model(model, to_file='model1.png')

history = model.fit(x=[X_train[:, 0:45], X_train[:, 45:]], y=Y_train, batch_size=500, epochs=5,
                    validation_data=([X_test[:, 0:45], X_test[:, 45:]], Y_test))
