import argparse
import math
import numpy as np
import pickle

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical

from sklearn.externals import joblib

from utils import *

NUM_SECONDS = 3
NUM_SAMPLES = 5

NUM_CHANNELS = 129
NUM_TIMEPOINTS = 1034
NUM_GENRES = 10

TEST_P = 0.2

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=int)
parser.add_argument('--samples', type=int)
args = parser.parse_args()

NUM_SECONDS = args.seconds
NUM_SAMPLES = args.samples

ARCH = 'img'

INPUT_FN = 'processed_data/xy_{}s_{}_img.pkl'.format(
    NUM_SECONDS, NUM_SAMPLES, ARCH)
MODEL_FN = 'models/{}s_{}_{}_model.h5'.format(NUM_SECONDS, NUM_SAMPLES, ARCH)
HISTORY_FN = 'models/{}s_{}_{}_history.pkl'.format(
    NUM_SECONDS, NUM_SAMPLES, ARCH)
FEATURES_FN = 'features/{}s_{}_{}_features.pkl'.format(
    NUM_SECONDS, NUM_SAMPLES, ARCH)

print("Saving to {}...".format(MODEL_FN))

# x, y = load_dataset('/multicomp/datasets/GTZAN', NUM_SECONDS, NUM_SAMPLES)
x, y = joblib.load(INPUT_FN)

p = np.random.permutation(len(x))
x = x[p]
y = y[p]

x_train = x[:int((1 - TEST_P) * len(x))]
y_train = y[:int((1 - TEST_P) * len(y))]

x_test = x[int((1 - TEST_P) * len(x)):]
y_test = y[int((1 - TEST_P) * len(y)):]

model = Sequential()

model.add(Convolution2D(nb_filter=16, nb_row=9, nb_col=9,
                        activation='relu', border_mode='same',
                        input_shape=(1, NUM_CHANNELS, NUM_TIMEPOINTS)))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Convolution2D(nb_filter=16, nb_row=9, nb_col=9,
                        activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(output_dim=100, activation='relu', name='final_features'))

model.add(Dropout(0.5))

model.add(Dense(output_dim=NUM_GENRES, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(x=x_train, y=to_categorical(y_train),
                    validation_data=(x_test, to_categorical(y_test)),
                    batch_size=32, nb_epoch=50, verbose=1)

pred = model.predict_classes(x=x_test, batch_size=32, verbose=1)
acc = np.mean(y_test == pred)
print("Accuracy: {}".format(acc))

model.save_weights(MODEL_FN)
with open(HISTORY_FN, 'wb') as outfile:
    pickle.dump(history.history, outfile)

style_model = Model(input=model.input,
                    output=model.get_layer('final_features').output)

style_features = style_model.predict(x_test)
with open(FEATURES_FN, 'wb') as outfile:
    pickle.dump((style_features, y_test), outfile)
