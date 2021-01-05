# Piven

This is an implementation of the model described in the following paper:

> Simhayev, Eli, Gilad Katz, and Lior Rokach. "PIVEN: A Deep Neural Network for Prediction Intervals with Specific Value Prediction." arXiv preprint arXiv:2006.05139 (2020).

I have copied some of the code from the [paper's code base](https://github.com/elisim/piven), and cite the author's paper where this is the case.

<figure>
    <img src="https://github.com/elisim/piven/blob/master/piven_architecture.jpg" height=250 width=800>
    <figcaption>Piven layer, from from Simhayev, Gilad and Rokach (2020).</figcaption>
</figure>

## In short

A neural network with a Piven (Prediction Intervals with specific value prediction) output layer returns a point
prediction as well as a lower and upper prediction interval (PI) for each target in a regression problem. 

This is useful because it allows you to quantify uncertainty in the point-predictions.

## Using Piven

Using the piven module is quite straightforward. For a simple MLP with a piven output, you can run:

```python
import numpy as np
from piven.models import PivenMlpModel
from sklearn.preprocessing import StandardScaler
# Make some data
seed = 26783
np.random.seed(seed)
# create some data
n_samples = 500
x = np.random.uniform(low=-2.0, high=2.0, size=(n_samples, 1))
y = 1.5 * np.sin(np.pi * x[:, 0]) + np.random.normal(
    loc=0.0, scale=1 * np.power(x[:, 0], 2)
)
x_train = x[:400, :].reshape(-1, 1)
y_train = y[:400]
x_valid = x[400:, :].reshape(-1, 1)
y_valid = y[400:]

# Build piven model
model = PivenMlpModel(
        input_dim=1,
        dense_units=(64, 64),
        dropout_rate=(0.1, 0.1),
        lambda_=25.0,
        bias_init_low=-3.0,
        bias_init_high=3.0,
        lr=0.0001,
)
# Normalize input data
model.build_model(preprocess=StandardScaler())
# You can pass any arguments that you would also pass to a keras model
model.fit(x_train, y_train, model__epochs=200, model__validation_split=.2)
model.score()
```

You can score the model by calling the `score()` method:

```python
y_pred, y_ci_low, y_ci_high = model.predict(x_test, return_prediction_intervals=True)
model.score(y_true, y_pred, y_ci_low, y_ci_high)
```

To persist the model on disk, call the `save()` method:

```python
model.save("path-to-model-folder", model=True, predictions=True)
```

This will save the metrics, keras model, and model predictions to the folder.

For additional examples, see the 'tests' and 'notebooks' folders.

## Creating your own model with Piven layer

You can use a Piven layer on any neural network architecture. The authors of the Piven paper use it on top of
a bunch of [CNN layers](https://github.com/elisim/piven/blob/master/imdb/main.py) to predict people's age.

Suppose that you want to create an Model with a Piven output layer. Because this module uses the 
[KerasRegressor](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasRegressor)  wrapper 
from the tensorflow library to make scikit-compatible keras models, you would first specify a build
function like so:

```python
import tensorflow as tf
from piven.layers import Piven
from piven.metrics.tensorflow import picp, mpiw
from piven.loss import piven_loss


def piven_model(input_size, hidden_units):
    i = tf.keras.layers.Input((input_size,))
    x = tf.keras.layers.Dense(hidden_units)(i)
    o = Piven()(x)
    model = tf.keras.models.Model(inputs=i, outputs=o)
    model.compile(optimizer="rmsprop", metrics=[picp, mpiw], 
                  loss=piven_loss(lambda_in=15.0, soften=160.0, 
                  alpha=0.05))
    return model
```

The most straightforward way of running your Model is to subclass the `PivenBaseModel` class. This requires you
to define a `build_model()` and `load()` method. In the former, you specify how the model should be defined. In the latter,
you specify how the model should be loaded from disk. In practice, this will always look the same, but you need to pass the
model build function.

```python
from piven.models.base import PivenBaseModel
from piven.scikit_learn.wrappers import PivenKerasRegressor
from piven.scikit_learn.compose import PivenTransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


class MyPivenModel(PivenBaseModel):
    def build_model():
        model = PivenKerasRegressor(build_fn=piven_model, **self.params)
        # Finally, normalize the output target
        self.model = PivenTransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
        return self

    @classmethod
    def load(cls, path: str):
        model_config = MyPivenModel.load_model_config(path)
        model = MyPivenModel.load_model_from_disk(build_fn=piven_model, path)
        run = cls(**Model_config)
        run.model = model
        return run
```

To initialize the model, call:

```python
MyPivenModel(
    input_size=3,
    hidden_units=32
)
```

Note that the inputs to `MyPivenModel` must match the inputs to the `piven_model` function.

You can now call all methods defined as in the PivenBaseModel class. Check the 
[PivenMlpModel class](https://gitlab.com/jasperginn/piven.py/-/blob/dev/src/piven/Models/mlp_regressor.py)
for a more detailed example.
