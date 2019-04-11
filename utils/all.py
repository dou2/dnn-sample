import numpy as np


# 读取文件，默认使用utf-8编码
def read_file_as_string(path, encoding='utf-8'):
    with open(file=path, mode="r", encoding=encoding) as fp:
        return fp.read()


# 按行读取文件，默认使用utf-8编码
def read_file_by_line(path, encoding='utf-8'):
    return read_file_as_string(path, encoding=encoding).splitlines()


# 加载bunch对象
def load_object(path):
    import pickle
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


# 写入bunch对象
def dump_object(path, obj):
    import pickle
    with open(path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def write_file_by_line(path, s, encoding='utf-8'):
    with open(file=path, mode="w", encoding=encoding) as fp:
        return fp.write(s)


# 结巴分词
def seg(s):
    import jieba
    return " ".join(jieba.cut(s, cut_all=False))


def load_arff(path):
    import arff
    with open(file=path, mode="r") as fp:
        a = arff.load(fp)
        b = np.array(a.get('data'))
    return b


def split_dataset(_data, train_size=0.9):
    np.random.shuffle(_data)
    m_samples, n_features = _data.shape
    print(_data.shape)
    num_train = int(m_samples * train_size)
    x_train = _data[0:num_train, 1:n_features - 1]
    y_train = _data[0:num_train, n_features - 1:n_features]
    x_test = _data[num_train:, 1:n_features - 1]
    y_test = _data[num_train:, n_features - 1:n_features]
    return x_train, y_train, x_test, y_test


# 加载csv格式数据集
def load_dataset(csv_file, train_size=0.9):
    import pandas as pd
    data = pd.read_csv(csv_file)
    return split_dataset(data.as_matrix(), train_size)


def split_dataset1(_data, train_size=0.9):
    np.random.shuffle(_data)
    m_samples, n_features = _data.shape
    print(_data.shape)
    num_train = int(m_samples * train_size)
    x_train = _data[0:num_train, 0:n_features - 1]
    y_train = _data[0:num_train, n_features - 1:n_features]
    x_test = _data[num_train:, 0:n_features - 1]
    y_test = _data[num_train:, n_features - 1:n_features]
    return x_train, y_train, x_test, y_test


# 加载csv格式数据集
def load_dataset1(csv_file, train_size=0.9):
    import pandas as pd
    data = pd.read_csv(csv_file)
    return split_dataset1(data.as_matrix(), train_size)


def save_model(session, export_dir):
    import tensorflow as tf
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(
        session,
        [tf.saved_model.tag_constants.SERVING]
    )
    builder.save()
