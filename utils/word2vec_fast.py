import numpy as np
from sklearn.externals import joblib

_unknown_word = '<UNK>'


class Word2VecFast:
    def __init__(self, vocabulary, word_embeddings):
        self._word_embeddings = word_embeddings
        self._vocabulary = vocabulary

    def word2vec(self, word):
        if word in self._vocabulary:
            p = self._vocabulary[word]
            return self._word_embeddings[p]
        else:
            return np.zeros(shape=self._word_embeddings.shape[1], dtype=np.float32)

    def words_id(self, sentence):
        wid = []
        ss = sentence.split()
        for s in ss:
            if s in self._vocabulary:
                wid.append(self._vocabulary[s])
            else:
                wid.append(self._vocabulary[_unknown_word])
        return wid

    def word_embeddings(self):
        return self._word_embeddings

    def word_shape(self):
        return self._word_embeddings.shape

    # 词向量处理
    @staticmethod
    def load_word2vec_format(file_path, accept_words=None, word_shape=None):
        vocabulary = {}
        word_embeddings = []
        with open(file=file_path, encoding='utf8') as file:
            for line_num, line in enumerate(file):
                values = line.split()
                word = values[0]
                if accept_words is None or word in accept_words:
                    vec = np.asarray(values[1:], dtype='float32')
                    word_embeddings.append(vec)
                    vocabulary[word] = line_num
            vocabulary[_unknown_word] = line_num
            word_embeddings.append(np.zeros(shape=word_shape, dtype=np.float32))
        return Word2VecFast(vocabulary, np.asarray(word_embeddings))

    def save(self, file_path):
        obj = dict()
        obj['word_embeddings'] = self._word_embeddings
        obj['vocabulary'] = self._vocabulary
        joblib.dump(obj, file_path)

    @staticmethod
    def load(file_path):
        o = joblib.load(filename=file_path)
        return Word2VecFast(vocabulary=o['vocabulary'], word_embeddings=o['word_embeddings'])
