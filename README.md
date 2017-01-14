# multicomp-hackathon

## Generating and inverting the spectrograms

We generate spectograms with Short-time Fourier Transform (STFT) and processing it as `S = log(1 + |STFT(x)|)` where `x` is the signal and `S` is the spectogram. This gives us, essentially, the intensity of various frequencies at different time points. This is done with librosa as follows.
```python
x, sr = librosa.load(audio_file)
D = librosa.stft(x)
S = np.log1p(np.abs(D))
```

To invert the spectograms, we use the [Griffin-Lim algorithm](http://cbcl.mit.edu/publications/ps/signalrec_ICSLP06.pdf) which iteratively estimates the phase information that was lost by taking magnitude for the spectogram.
```python
x = np.exp(S) - 1
p = 2 * np.pi * np.random.random_sample(x.shape) - np.pi # Start with a random estimate
for i in range(500): # Do 500 iterations
	Q = x * np.exp(1j*p) # Estimate the full STFT using the magnitude and the phase estimate
	y = librosa.istft(Q) + 1e-6 # invert the STFT
	p = np.angle(librosa.stft(y)) # Improve the phase estimate using the new signal
```

To generate our dataset, we generate spectograms from multiple 5 second samples from each audio file.

## The classification model

For building the classification model, we use [Keras](http://keras.io), with [Theano](http://deeplearning.net/software/theano/) as a back-end.

...

We first initialize a `Sequential` model object, since our classification model consists of a simple linear stack of layers.

```python
from keras.models import Sequential

model = Sequential()
```

Now we have our empty model, and we need to add our layers to the model one by one using `.add`.

> fill in why we chose the architecture that we did

```python
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

model.add(Convolution2D(nb_filter=32, nb_row=1, nb_col=9,
                        activation='relu', border_mode='same',
                        input_shape=(NUM_CHANNELS, 1, NUM_TIMEPOINTS)))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Convolution2D(nb_filter=32, nb_row=1, nb_col=9,
                        activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(1, 2)))

model.add(Flatten())
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
