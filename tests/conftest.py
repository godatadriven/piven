from typing import Union, Callable
import pytest
import numpy as np
from piven.scikit_learn.wrappers import PivenRegressor
from piven.scikit_learn.compose import PivenTransformedTargetRegressor
from piven.metrics.tensorflow import mpiw, picp
from piven.loss import piven_loss
from piven.regressors import build_keras_piven
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
def keras_model_function() -> Callable:
    def keras_model(input_size, hidden_units=(128, 128)):
        m = build_keras_piven(
            input_dim=input_size,
            dense_units=hidden_units,
            dropout_rate=(0.0, 0.0),
            activation="relu",
        )
        m.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0007),
            loss=piven_loss(25.0, 160.0, 0.05),
            metrics=[picp, mpiw],
        )
        return m

    return keras_model


@pytest.fixture(scope="function")
def piven_model_wrapper(keras_model_function: Callable) -> PivenRegressor:
    return PivenRegressor(
        build_fn=keras_model_function, input_size=1, hidden_units=(128, 128)
    )


@pytest.fixture(scope="function")
def piven_model_pipeline(piven_model_wrapper: PivenRegressor) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", piven_model_wrapper)])


@pytest.fixture(scope="function")
def transformed_piven_regressor(
    piven_model_pipeline: Pipeline
) -> PivenTransformedTargetRegressor:
    return PivenTransformedTargetRegressor(
        regressor=piven_model_pipeline, transformer=StandardScaler()
    )
