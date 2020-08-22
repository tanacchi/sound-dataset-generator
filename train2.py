import numpy as np
from pydub import AudioSegment
from glob import glob
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, Sequential
from keras.layers import Dense,  Flatten, InputLayer
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers import Activation
from keras.optimizers import Adam
import os


def wav_to_array(file):
    song = AudioSegment.from_wav(file)
    song_data = song._data
    song_arr = np.frombuffer(song_data, np.int16)
    return song_arr

import time
files = list(glob("./output/*.wav"))[::4]
X = np.array([wav_to_array(file) for file in files])
X = X.reshape((X.shape[0], X.shape[1], 1))

filenames = [os.path.basename(file) for file in files]
labels = [int(filename[0]) for filename in filenames]
print(labels)
print(len(files))
outputs = 2
y = np.array([np.identity(outputs)[l] for l in labels])
print(X.shape)
print(y.shape)
print(y)


random_state = 42
train_X, test_X = X[50:], X[:50]
train_y, test_y = y[50:], y[:50]
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=random_state)

features = train_X.shape[1]

batch_size = 50
model = Sequential()
model.add(InputLayer(input_shape=(features, 1), name='x_inputs'))
model.add(LSTM(256,
              input_shape=(1, features),
              batch_size=batch_size,
              #output_shape=(None, dims),
              return_sequences=True,
              activation='tanh'))
model.add(LSTM(256, return_sequences=True, activation="sigmoid"))
model.add(LSTM(256, return_sequences=True, activation="selu"))
model.add(Dense(256))
model.add(Dense(outputs))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_X, train_y, batch_size=batch_size, epochs=50)

from sklearn.metrics import roc_auc_score
pred_y_x1 = model.predict(test_X, batch_size=batch_size)
print(pred_y_x1)
print(roc_auc_score(test_y, pred_y_x1))
