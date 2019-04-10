import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

labeled_images = pd.read_csv('D:/kaggle/data/Digit_Recognizer/train.csv')
images = labeled_images.iloc[0:1000, 1:]
labels = labeled_images.iloc[0:1000, :1]
images = images / 255.
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)

clf = svm.SVC(max_iter=10000)
clf.fit(train_images, train_labels.values.ravel())
score = clf.score(test_images, test_labels)
print(score)

test_data = pd.read_csv('D:/kaggle/data/Digit_Recognizer/test.csv')
test_data = test_data / 255.
results = clf.predict(test_data[0:50])
print(results)
