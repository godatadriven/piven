import pytest
import numpy as np


@pytest.fixture(scope="module")
def mock_data():
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
    y_train = np.stack((y_train, y_train), axis=1)  # make this 2d so will be accepted
    y_valid = np.stack((y_valid, y_valid), axis=1)
    return x_train, x_valid, y_train, y_valid
