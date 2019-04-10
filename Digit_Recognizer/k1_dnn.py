import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

labeled_images = pd.read_csv('D:/kaggle/data/Digit_Recognizer/train.csv')
images = labeled_images.iloc[0:40000, 1:]
labels = labeled_images.iloc[0:40000, :1]
images = images / 255.
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
num_classes = Y_train.shape[1]
print(num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(X_train, Y_train, batch_size=500, epochs=10, validation_data=(X_test, Y_test))
