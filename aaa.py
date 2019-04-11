import keras
import pandas as pd

from basic_classification import MultiClassify
from utils.all import *

a = pd.read_csv('d:/t_contract_paragraph1.csv')
# b = a[a['N'] == 1]
# c = b.copy()
# for i in range(40):
#     a = pd.concat([a, c])
# f = a[a['N'] == 1]
# print(a.describe())

data = a.as_matrix()
x_train, y_train, x_test, y_test = split_dataset1(data, train_size=0.7)
n_features = x_train.shape[1]

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

a = MultiClassify(n_features, 2)
a.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test), export_dir='d:/tmp/t_contract_paragraph11')
