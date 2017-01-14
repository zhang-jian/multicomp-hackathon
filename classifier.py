import math
import pickle

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.utils.np_utils import to_categorical

from sklearn.externals import joblib

from utils import *

NUM_SECONDS = 3
NUM_CHANNELS = 1025
NUM_GENRES = 10

NUM_TIMEPOINTS = math.ceil(43.1 * NUM_SECONDS)

MODEL_FN = 'model.h5'
HISTORY_FN = 'history.pkl'

# x, y = load_dataset('/multicomp/datasets/GTZAN', NUM_SECONDS, 5)
x, y = joblib.load('xy_3s_5.pkl')

model = Sequential()

model.add(Convolution2D(nb_filter=32, nb_row=1, nb_col=9,
                        activation='relu', border_mode='same',
                        input_shape=(NUM_CHANNELS, 1, NUM_TIMEPOINTS)))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Convolution2D(nb_filter=32, nb_row=1, nb_col=9,
                        activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Flatten())
model.add(Dense(output_dim=NUM_GENRES, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

history = model.fit(x=x, y=to_categorical(y), batch_size=32, nb_epoch=10,
                    verbose=1, validation_split=0.2, shuffle=True)
pred = model.predict_classes(x=x, batch_size=32, verbose=1)
acc = np.mean(y == pred)
print("Accuracy: {}".format(acc))

model.save_weights(MODEL_FN)
with open(HISTORY_FN, 'wb') as outfile:
    pickle.dump(history.history, outfile)
