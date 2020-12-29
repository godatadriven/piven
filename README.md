# Piven

This is an implementation of the model described in the following paper:

> Simhayev, Eli, Gilad Katz, and Lior Rokach. "PIVEN: A Deep Neural Network for Prediction Intervals with Specific Value Prediction." arXiv preprint arXiv:2006.05139 (2020).

I have copied some of the code from the [paper's code base](https://github.com/elisim/piven), and cite the author's paper where this is the case.

## In short

A neural network with a Piven (Prediction Intervals with specific value prediction) output layer returns a point
prediction as well as a lower and upper prediction interval (PI) for each target in a regression problem. 

This is useful because it allows you to quantify uncertainty in the point-predictions.

## Using Piven

Using the piven module is quite straightforward. For a simple MLP with a piven output, you can run:

```python
import numpy as np
from piven.experiments import PivenMlpExperiment
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
model = PivenMlpExperiment(
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

## Creating your own Piven model

You can use a Piven layer on any neural network architecture. REF REPO AUTHORS example convo network.

This library uses the KerasRegressor wrapper from the tensorflow library to make scikit-compatible
keras models.  

Suppose that you want to create an experiment with a Piven output layer. You would first specify a build
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
```

Then, you can create an experiment as follows:

```python
from piven.experiments.base import PivenBaseExperiment
from piven.scikit_learn.wrappers import PivenKerasRegressor
from piven.scikit_learn.compose import PivenTransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


class MyPivenExperiment(PivenBaseExperiment):
    def build_model():
        model = PivenKerasRegressor(build_fn=piven_model, **self.params)
        # Finally, normalize the output target
        self.model = PivenTransformedTargetRegressor(
            regressor=model, transformer=StandardScaler()
        )
        return self

    @classmethod
    def load(cls, path: str):
        experiment_config = MyPivenExperiment.load_experiment_config(path)
        model = MyPivenExperiment.load_model_from_disk(build_fn=piven_model, path)
        run = cls(**experiment_config)
        run.model = model
        return run
```

You can now call all methods defined as in the PivenBaseExperiment class. Check the PivenMlpExperiment class
for a more detailed example. ADD LINK.


