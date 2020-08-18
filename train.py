import numpy as np
from pydub import AudioSegment
from glob import glob
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import Dense,  Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
import os


def wav_to_array(file):
    song = AudioSegment.from_wav(file)
    song_data = song._data
    song_arr = np.frombuffer(song_data, np.int16)
    return song_arr

import time
files = list(glob("./output/*.wav"))[::5]
data_size = wav_to_array(files[0]).shape[0]
file_batch = 50
X = np.zeros((0, data_size, 1))
for i in range(0, len(files), file_batch):
    mini_files = files[i: i+file_batch]
    x = [wav_to_array(file) for file in mini_files]
    x = list(filter(lambda x: x.shape[0] == data_size, x))
    x = np.array(x)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    X = np.concatenate([X, x])


filenames = [os.path.basename(file) for file in files]
print(filenames[0])
print(filenames[0][0])
labels = [int(filename[0]) for filename in filenames]
print(labels)
y = np.array(labels)
print(len(files))
print(X.shape)
print(y.shape)
outputs = 1


random_state = 42
train_X, test_X = X[50:], X[:10]
train_y, test_y = y[50:], y[:10]
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=random_state)

features = train_X.shape[1]

x_inputs = Input(shape=(features, 1), name='x_inputs') # (特徴量数, チャネル数)
x = Conv1D(128, 256, strides=256,
           padding='valid', activation='relu') (x_inputs)
x = Conv1D(32, 8, activation='relu') (x) # (チャネル数, フィルタの長さ )
x = MaxPooling1D(4) (x) # （フィルタの長さ）
x = Conv1D(32, 8, activation='relu') (x)
x = MaxPooling1D(4) (x)
x = Conv1D(32, 8, activation='relu') (x)
x = MaxPooling1D(4) (x)
x = Conv1D(32, 8, activation='relu') (x)
x = MaxPooling1D(4) (x)
x = Flatten() (x)
x = Dense(100, activation='relu') (x) #（ユニット数）
x_outputs = Dense(outputs, activation='sigmoid', name='x_outputs') (x)

model = Model(inputs=x_inputs, outputs=x_outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_y, batch_size=600, epochs=50)

from sklearn.metrics import roc_auc_score
pred_y_x1 = model.predict(test_X, batch_size=50)
print(pred_y_x1)
print(roc_auc_score(test_y, pred_y_x1))
