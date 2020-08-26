import random
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import glob
import sys
from pprint import pprint

mode = sys.argv[1]
length = sys.argv[2]

parent = f"train_data/{mode}/*{length}*"
X_train = []
y_train = []
parents = glob.glob(parent)
for parent in parents:
    path = parent + "/*/*"
    point = 0
    # if "pokemon" in parent:
    #     point = 0
    # else:
    #     point = 1
    paths = glob.glob(path)
    # pprint(paths)
    for path in paths:
        img = cv2.imread(path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        X_train.append(img)
        point = 0 if "pokemon" in path else 1
        y_train.append(point)
        print(path, point)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=False)
X_train = np.array(X_train)
# datagen.fit(X_train)
y_train = np.array(y_train)
y_train = y_train.T
y_train = to_categorical(y_train)
# print(X_train)
# print(y_train)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.3,shuffle=True,random_state=200)
# print(y_train)
# print(y_test)
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.models import Model, Sequential
model = Sequential()
model.add(Conv2D(input_shape=(800, 800, 3), filters=32,kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])
batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, epochs=40)
# model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
#                     steps_per_epoch=len(X_train) / batch_size, epochs=50)
score = model.evaluate(X_test, y_test, verbose=1)
print()
print("Test loss:", score[0])
print("Test accuracy:", score[1])
y_pred = model.predict(X_test)
# y_pred = np.argmax(y_pred, axis=1)
# y_pred = np.where(y_pred > 0.5, 1, 0)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
sns.heatmap(cm,square=True,annot=True,cbar=False,fmt='d',cmap='RdPu')
plt.xlabel('predicted class')
plt.ylabel('true value')
plt.show()
# print('confusion matrix = \n', confusion_matrix(y_true=y_test, y_pred=y_pred))
# confusion_matrix(y_true, y_pred)
print(y_pred)
