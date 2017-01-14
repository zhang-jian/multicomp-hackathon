# multicomp-hackathon

## Generating the spectrograms

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

Finally, we need to configure the learning process of the model, using `.compile`. Here we choose to use the [Adam](https://arxiv.org/abs/1412.6980) optimizer because...

> justify optimizer/loss function choices

We would like to see the model accuracy after each batch, so we add `metrics=['accuracy']`.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Now we've finally finished constructing the model; it's time to start training. For this we use `.fit`, where `x` is our training features, and `to_categorical(y)` is our training labels (cast to a categorical variable, so that we can use the categorical cross-entropy loss function). We use a batch size of 32 samples (`batch_size=32`) over 10 epochs (`nb_epoch=10`). We split our data into a 80% training, 20% validation set with `validation_split=0.2`, and we shuffle our data ordering (`shuffle=True`). Finally, we want to see our performance along the way (`verbose=1`).

```python
history = model.fit(x=x, y=to_categorical(y), batch_size=32, nb_epoch=10,
                    verbose=1, validation_split=0.2, shuffle=True)
```
