from typing import Union
import pytest
import numpy as np
from piven.wrappers import PivenModelWrapper
from piven.transformers import PivenTransformedTargetRegressor
from piven.metrics import picp, mpiw
from piven.layers import Piven
from piven.loss import piven_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


@pytest.fixture(scope="module")
def mock_data() -> Union[np.array, np.array, np.array, np.array]:
    # fix seed
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
    return x_train, x_valid, y_train, y_valid


@pytest.fixture(scope="function")
def keras_model_function() -> tf.python.keras.engine.functional.Functional:
    def keras_model(input_size, dropout_rate):
        i = tf.keras.layers.Input(shape=(input_size,))
        x = tf.keras.layers.Dense(128)(i)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        o = Piven()(x)
        m = tf.keras.models.Model(inputs=i, outputs=[o])
        m.compile(
            optimizer="adam",
            loss=piven_loss(True, 15.0, 160.0, 0.05),
            metrics=[picp, mpiw],
        )
        return m

    return keras_model


@pytest.fixture(scope="function")
def piven_model_wrapper(keras_model_function) -> PivenModelWrapper:
    return PivenModelWrapper(build_fn=keras_model_function(1, 0.1))


@pytest.fixture(scope="function")
def piven_model_pipeline(piven_model_wrapper) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", piven_model_wrapper)])


@pytest.fixture(scope="function")
def transformed_piven_regressor(
    piven_model_pipeline
) -> PivenTransformedTargetRegressor:
    return PivenTransformedTargetRegressor(
        regressor=piven_model_pipeline, transformer=StandardScaler()
    )
