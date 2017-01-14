import argparse
import math

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

import theano
import theano.tensor as T

from utils import *


def get_adam_updates(f, params, lr=1, b1=0.9, b2=0.999, e=1e-8, dec=0, norm_grads=False):
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
ARCH = 'img_a'

NUM_CHANNELS = 257
NUM_TIMEPOINTS = 257
NUM_GENRES = 10

# OPTIMIZATION_ITERATIONS = 1
OPTIMIZATION_ITERATIONS = 3000

LAMBDA = 1.

# add commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, required=True)
parser.add_argument('--style', type=str, required=True)
parser.add_argument('--lambdav', type=float, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

LAMBDA = args.lambdav

MODEL_FN = 'models/{}s_{}_{}_model.h5'.format(NUM_SECONDS, NUM_SAMPLES, ARCH)
print("Loading {}...".format(MODEL_FN))

PCA_FN = 'features/{}s_{}_img_pca.pkl'.format(NUM_SECONDS, NUM_SAMPLES)
pca, xbar = joblib.load(PCA_FN)

pca = pca.astype('float32')
xbar = xbar.astype('float32')

# load content and style spectrogram
input_content_S = load_audio_get_spectrogram(
    args.content, duration=NUM_SECONDS)
input_style_S = load_audio_get_spectrogram(
    args.style, duration=NUM_SECONDS)

# initializing a blank spectrogram
S_init = np.random.normal(
    size=(1, 2617), scale=10).astype('float32')
# must be a shared variable because it needs to be mutable
S = theano.shared(S_init)

S_pca = xbar + T.dot(S, pca)
S_pca = T.reshape(S_pca, (1, 1, 257, 257))

# wrapper for spectrogram for use with keras
input_S = Input(tensor=S_pca, shape=(1, NUM_CHANNELS,
                                     NUM_TIMEPOINTS), dtype='float32')

# apply each layer to the tensor
z = Convolution2D(nb_filter=16, nb_row=5, nb_col=5,
                  activation='relu', border_mode='same')(input_S)
z = MaxPooling2D(pool_size=(2, 2))(z)

z = Convolution2D(nb_filter=16, nb_row=5, nb_col=5,
                  activation='relu', border_mode='same')(z)
z = MaxPooling2D(pool_size=(2, 2))(z)

z = Flatten()(z)
style_features_S = Dense(output_dim=100, activation='relu')(z)
output = Dense(output_dim=NUM_GENRES,
               activation='softmax')(style_features_S)

# wrapping as keras model
model = Model(input=input_S, output=output)

# load weights from style_classifier run
model.load_weights(MODEL_FN)

# building the loss function
get_style_features = theano.function([S_pca], style_features_S)
# S.set_value(input_style_S.reshape(1, 1, NUM_CHANNELS, NUM_TIMEPOINTS))

input_style_features = get_style_features(
    input_style_S.reshape(1, 1, 257, 257))
# S.set_value(S_init)
# S.set_value(load_audio_get_spectrogram(
#     '/multicomp/datasets/GTZAN/metal/metal.00081.au', duration=NUM_SECONDS).reshape(1, 1, NUM_CHANNELS, NUM_TIMEPOINTS))

style_loss = T.sum(T.square(style_features_S - input_style_features))
content_loss = T.sum(
    T.square(S_pca - input_content_S.reshape(1, 1, NUM_CHANNELS, NUM_TIMEPOINTS)))
total_loss = content_loss + LAMBDA * style_loss
# total_loss = style_loss

optim_step = theano.function(
    [], total_loss, updates=get_adam_updates(total_loss, [S]))

with tqdm(desc="Optimizing...", ncols=80, ascii=False,
          total=OPTIMIZATION_ITERATIONS) as bar:

    for i in range(OPTIMIZATION_ITERATIONS):
        current_loss = optim_step().item()
        bar.set_description(
            "Optimizing... (loss: {:0.4g})".format(current_loss))
        bar.update(1)

get_pcaS = theano.function([], S_pca)
output_S = get_pcaS()

output_S = np.clip(output_S[0, 0, :, :], -50, 50)
convert_spectrogram_and_save(output_S, args.output)

output_style_features = get_style_features(
    output_S.reshape(1, 1, NUM_CHANNELS, NUM_TIMEPOINTS))
joblib.dump((input_style_features, output_style_features),
            args.output[:-4] + '.pkl')
