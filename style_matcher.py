import argparse
import math

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

import theano
import theano.tensor as T

from utils import *


def get_adam_updates(f, params, lr=1e-3, b1=0.9, b2=0.999, e=1e-8, dec=5e-3, norm_grads=False):
    """Generate updates to optimize using the Adam optimizer with linear learning rate decay."""
    t = theano.shared(0)
    ms = [theano.shared(np.zeros(param.shape.eval(),
                                 dtype=floatX), borrow=True) for param in params]
    vs = [theano.shared(np.zeros(param.shape.eval(),
                                 dtype=floatX), borrow=True) for param in params]

    gs = T.grad(f, params)
    if norm_grads:
        gs = [g / (T.sum(T.abs_(g)) + 1e-8) for g in gs]
    t_u = (t, t + 1)
    m_us = [(m, b1 * m + (1. - b1) * g) for m, g in zip(ms, gs)]
    v_us = [(v, b2 * v + (1. - b2) * T.sqr(g)) for v, g in zip(vs, gs)]
    t_u_f = T.cast(t_u[1], floatX)
    lr_hat = (lr / (1. + t_u_f * dec)) * T.sqrt(1. -
                                                T.pow(b2, t_u_f)) / (1. - T.pow(b1, t_u_f))
    param_us = [(param,  param - lr_hat * m_u[1] / (T.sqrt(v_u[1]) + e))
                for m_u, v_u, param in zip(m_us, v_us, params)]
    return m_us + v_us + param_us + [t_u]

NUM_SECONDS = 3
NUM_SAMPLES = 5
NUM_CHANNELS = 1025
NUM_GENRES = 10

# OPTIMIZATION_ITERATIONS = 1
OPTIMIZATION_ITERATIONS = 1000

LAMBDA = 1.

# add commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, required=True)
parser.add_argument('--style', type=str, required=True)
parser.add_argument('--lambdav', type=float, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

LAMBDA = args.lambdav

MODEL_FN = 'models/{}s_{}_model.h5'.format(NUM_SECONDS, NUM_SAMPLES)
print("Loading {}...".format(MODEL_FN))
NUM_TIMEPOINTS = int(math.ceil(43.1 * NUM_SECONDS))

# load content and style spectrogram
input_content_S = load_audio_get_spectrogram(
    args.content, duration=NUM_SECONDS)
input_style_S = load_audio_get_spectrogram(
    args.style, duration=NUM_SECONDS)

# initializing a blank spectrogram
S_init = np.zeros((1, NUM_CHANNELS, 1, NUM_TIMEPOINTS), dtype='float32')
# must be a shared variable because it needs to be mutable
S = theano.shared(S_init)

# wrapper for spectrogram for use with keras
input_S = Input(tensor=S, shape=(NUM_CHANNELS, 1,
                                 NUM_TIMEPOINTS), dtype='float32')

# apply each layer to the tensor
z = Convolution2D(nb_filter=32, nb_row=1, nb_col=9,
                  activation='relu', border_mode='same')(input_S)
z = MaxPooling2D(pool_size=(1, 2))(z)

z = Convolution2D(nb_filter=64, nb_row=1, nb_col=9,
                  activation='relu', border_mode='same')(z)
z = MaxPooling2D(pool_size=(1, 2))(z)

# z = Dropout(0.5)(z)

style_features_S = Flatten()(z)
output = Dense(output_dim=NUM_GENRES,
               activation='softmax')(style_features_S)

# wrapping as keras model
model = Model(input=input_S, output=output)

# load weights from style_classifier run
model.load_weights(MODEL_FN)

# building the loss function
get_style_features = theano.function([], style_features_S)
S.set_value(input_style_S.reshape(1, NUM_CHANNELS, 1, NUM_TIMEPOINTS))

input_style_features = get_style_features()
S.set_value(S_init)

style_loss = T.sum(T.square(style_features_S - input_style_features))
content_loss = T.sum(
    T.square(S - input_content_S.reshape(1, NUM_CHANNELS, 1, NUM_TIMEPOINTS)))
total_loss = content_loss + LAMBDA * style_loss

optim_step = theano.function(
    [], total_loss, updates=get_adam_updates(total_loss, [S]))

with tqdm(desc="Optimizing...", ncols=80, ascii=False,
          total=OPTIMIZATION_ITERATIONS) as bar:

    for i in range(OPTIMIZATION_ITERATIONS):
        current_loss = optim_step().item()
        bar.set_description(
            "Optimizing... (loss: {:0.4g})".format(current_loss))
        bar.update(1)

output_S = np.clip(S.get_value()[0, :, 0, :], -50, 50)
convert_spectrogram_and_save(output_S, args.output)
