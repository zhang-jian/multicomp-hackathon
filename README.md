# multicomp-hackathon

## Problem definition

In our work we want to transfer the *style* of one piece of music to another. More specifically we want the newly generated music piece to have the *content* of the source piece but match the *style* of the target.

While music style is really difficult to define and quantify, we narrow down our definition to style = genre. So for example if we have a classical piano piece as the source and a jazz piece as the target, the result should be piano jazz, while keeping the original content recognizable.

This work is inspired by similar work on style transfer in images

## Approach

We believe that musical style and content can be captured by different representations of a musical piece.

For capturing content we choose to use spectrograms, an audio representation reliant on the Fourier Transform of the audio signal, and example of a spectrogram with an audio signal that resulted in it:
![Spectrogram example](http://oyro.no/fig/spektrum.png)

Fortunately, for us it is also an "semi-invertible" operation which allows us to manipulate the spectrogram directly and still be able to generate an audio file from it.

In order to capture style/genre we need a representation that captures such information. If we build a neural network based classifier to predict musical genre we can arrive at such a representation that clusters similar genres close together.

## Dataset

As one of the building blocks of our approach we need a music genre classifier. In order to build it we use the common dataset for the task - [GTZAN Genre Collection](http://marsyasweb.appspot.com/download/data_sets/)

GRZAN contains a 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks.

## Generating and inverting the spectrograms

We generate spectrograms with Short-time Fourier Transform (STFT) and processing it as `S = log(1 + |STFT(x)|)` where `x` is the signal and `S` is the spectrogram. This gives us, essentially, the intensity of various frequencies at different time points. This is done with librosa as follows.
```python
x, _ = librosa.load(audio_file)
D = librosa.stft(x) # D is a complex matrix with the magnitude and phase of the fourier transform
S = np.log1p(np.abs(D)) # Take the magnitude with abs (call it Z) and return log(1 + Z) as the spectrogram
```

To invert the spectrograms, we use the [Griffin-Lim algorithm](http://cbcl.mit.edu/publications/ps/signalrec_ICSLP06.pdf) which iteratively estimates the phase information that was lost by taking magnitude for the spectrogram.
```python
D = np.exp(S) - 1
p = 2 * np.pi * np.random.random_sample(x.shape) - np.pi # Start with a random estimate
for i in range(500): # Do 500 iterations
	Q = D * np.exp(1j*p) # Estimate the full STFT using the magnitude and the phase estimate
	y = librosa.istft(Q) + 1e-6 # invert the STFT
	p = np.angle(librosa.stft(y)) # Improve the phase estimate using the new signal
```

To generate our dataset, we generate spectrograms from multiple 5 second samples from each audio file.

## The classification model

For building the classification model, we use [Keras](http://keras.io), with [Theano](http://deeplearning.net/software/theano/) as a back-end.

We first initialize a `Sequential` model object, since our classification model consists of a simple linear stack of layers.

```python
from keras.models import Sequential

model = Sequential()
```

Now we have our empty model, and we need to add our layers to the model one by one using `.add`.

```python
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

model.add(Convolution2D(nb_filter=16, nb_row=5, nb_col=5,
                        activation='relu', border_mode='same',
                        input_shape=(1, NUM_CHANNELS, NUM_TIMEPOINTS)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(nb_filter=16, nb_row=5, nb_col=5,
                        activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(output_dim=100, activation='tanh', name='final_features'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=NUM_GENRES, activation='softmax'))

```

Finally, we need to configure the learning process of the model, using `.compile`. Here we choose to use the [Adam](https://arxiv.org/abs/1412.6980) optimizer (`optimizer='adam'`) and categorical cross-entropy loss function (`loss='categorical_crossentropy'`) because these options are considered to be the current state-of-the-art for this kind of problem. We would like to see the model accuracy after each batch, so we add `metrics=['accuracy']`.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Now we've finally finished constructing the model; it's time to start training. For this we use `.fit`, where `x` is our training features, and `to_categorical(y)` is our training labels (cast to a categorical variable, so that we can use the categorical cross-entropy loss function). We use a batch size of 32 samples (`batch_size=32`) over 10 epochs (`nb_epoch=10`). We split our data into a 80% training, 20% validation set with `validation_split=0.2`, and we shuffle our data ordering (`shuffle=True`). Finally, we want to see our performance along the way (`verbose=1`).

```python
history = model.fit(x=x, y=to_categorical(y), batch_size=32, nb_epoch=10,
                    verbose=1, validation_split=0.2, shuffle=True)
```
