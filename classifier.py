import math
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical

from sklearn.externals import joblib

from utils import *

NUM_SECONDS = 3
NUM_CHANNELS = 1025
NUM_GENRES = 10

TEST_P = 0.2

NUM_TIMEPOINTS = math.ceil(43.1 * NUM_SECONDS)

MODEL_FN = 'model.h5'
HISTORY_FN = 'history.pkl'

# x, y = load_dataset('/multicomp/datasets/GTZAN', NUM_SECONDS, 5)
x, y = joblib.load('processed_data/xy_3s_50.pkl')

p = np.random.permutation(len(x))
x = x[p]
y = y[p]

x_train = x[:int((1 - TEST_P) * len(x))]
y_train = y[:int((1 - TEST_P) * len(y))]

x_test = x[int((1 - TEST_P) * len(x)):]
y_test = y[int((1 - TEST_P) * len(y)):]

model = Sequential()

model.add(Dropout(p=0.7, input_shape=(NUM_CHANNELS, 1, NUM_TIMEPOINTS)))

model.add(Convolution2D(nb_filter=8, nb_row=1, nb_col=9,
                        activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(output_dim=NUM_GENRES, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

history = model.fit(x=x_train, y=to_categorical(y_train),
                    validation_data=(x_test, to_categorical(y_test)),
                    batch_size=32, nb_epoch=10, verbose=1)

pred = model.predict_classes(x=x_test, batch_size=32, verbose=1)
acc = np.mean(y_test == pred)
print("Accuracy: {}".format(acc))

model.save_weights(MODEL_FN)
with open(HISTORY_FN, 'wb') as outfile:
    pickle.dump(history.history, outfile)
