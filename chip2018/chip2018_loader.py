# 加载数据，存储为对象，返回(x_train, y_train),(x_test,y_test)
import sys

sys.path.append('..')
from utils.word2vec_fast import *
from keras.preprocessing import sequence
from sklearn.externals import joblib

# 加载词向量
wv = Word2VecFast.load_word2vec_format(file_path='../data/chip2018/char_embedding.txt', word_shape=300)
print('word_embedding shape: ', wv.word_shape())

# 参数
sequence_length = 45
embedding_size = 300
num_filters = 128
num_classes = 2
num_epoch = 10
batch_size = 128
filter_height = 3


# 加载问题，问题->词id
def load_question():
    m_len = 0
    question = {}
    with open(file='../data/chip2018/question_id.csv', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                continue
            ss = line.split(',')
            qid, qw, qc = ss[0], ss[1], ss[2]
            question[qid] = wv.words_id(qc)
            m_len = max(len(question[qid]), m_len)
    return question, m_len


def load_dataset(path, question_w_dict=None):
    if question_w_dict is None:
        question_w_dict = {}
    train_data = []
    with open(file=path, encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # if line_num == 0:
            #     continue
            ss = line.split(',')
            a1 = question_w_dict[ss[0]]
            a2 = question_w_dict[ss[1]]
            a1 = sequence.pad_sequences([a1], maxlen=sequence_length, padding='post', value=wv.words_id('<UNK>'))[0]
            a2 = sequence.pad_sequences([a2], maxlen=sequence_length, padding='post', value=wv.words_id('<UNK>'))[0]
            a = np.hstack((a1, a2))
            b = np.hstack((a, int(ss[2])))
            train_data.append(b)
    data_ = np.array(train_data)
    np.random.shuffle(data_)
    return data_


if __name__ == '__main__':
    # question_word = load_question()
    # dataset = load_dataset('../data/chip2018/train.csv', question_word)
    # print('data shape: ', dataset.shape)
    # m = dataset.shape[0]
    # n = dataset.shape[1]
    # train_set_ratio = 0.9
    # num_train = int(m * train_set_ratio)
    #
    # X_train = dataset[0:num_train, 0:n - 1]
    # Y_train = dataset[0:num_train, n - 1:]
    # X_test = dataset[num_train:, 0:n - 1]
    # Y_test = dataset[num_train:, n - 1:]
    #
    # obj = dict()
    # obj['X_train'] = X_train
    # obj['Y_train'] = Y_train
    # obj['X_test'] = X_test
    # obj['Y_test'] = Y_test
    # joblib.dump(obj, 'd:/chip2018.data1')

    question_word, sequence_length = load_question()
    train_dataset = load_dataset('d:/chip2018/train.txt', question_word)
    print('train_dataset shape:', train_dataset.shape)
    test_dataset = load_dataset('d:/chip2018/test.txt', question_word)
    print('test_dataset shape:', test_dataset.shape)
    n = train_dataset.shape[1]
    X_train = train_dataset[:, 0:n - 1]
    Y_train = train_dataset[:, n - 1:]
    X_test = test_dataset[:, 0:n - 1]
    Y_test = test_dataset[:, n - 1:]

    obj = dict()
    obj['X_train'] = X_train
    obj['Y_train'] = Y_train
    obj['X_test'] = X_test
    obj['Y_test'] = Y_test
    joblib.dump(obj, 'd:/chip2018.data3')
