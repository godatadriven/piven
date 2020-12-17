import pathlib
import json
from typing import Tuple, List, Union
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from piven.loss import build_keras_piven, piven_loss
from sklearn.exceptions import NotFittedError

import tensorflow as tf
from piven.metrics import picp, mpiw
from piven.layers import Piven

# Dump custom metrics, loss and layers
# Need to do this when saving models.
tf.keras.utils.get_custom_objects().update(
    {"picp": picp, "mpiw": mpiw, "piven_loss": piven_loss, "Piven": Piven}
)


class PivenRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim: int,
        optimizer,
        dense_units: Tuple[int, ...] = (64,),
        dropout_rate: Tuple[float, ...] = (0.1,),
        activation: str = "relu",
        epochs: int = 100,
        batch_size: int = 128,
        validation_split: float = 0.1,
        verbose: bool = False,
        callbacks: List = None,
        metrics: List = None,
        reg=25.0,
        soften=160.0,
        alpha=0.05,
    ):
        super().__init__()
        """Build a keras MLP model and turn it into an sklearn-compliant model."""
        if len(dense_units) != len(dropout_rate):
            raise ValueError(
                "'dense_units' and 'dropout_rate' must have the same length"
                + f" (dense units = {dense_units}, "
                + f"dropout_rate = {dropout_rate})"
            )
        if not all(map(lambda x: isinstance(x, float), dropout_rate)):
            raise TypeError("All values in 'dropout_rate' must be floats.")
        if not all(map(lambda x: 0 <= x <= 1, dropout_rate)):
            raise ValueError("All values in 'dropout_rate' must be between 0 and 1.")
        if not all(map(lambda x: isinstance(x, int), dense_units)):
            raise TypeError("All values in 'dense_units' must be integers.")
        if not all(map(lambda x: x > 0, dense_units)):
            raise ValueError("All values in 'dense_units' must be larger than 0.")
        # Todo: check if each element in list is callback
        if not (isinstance(callbacks, list) or callbacks is None):
            raise ValueError("'callbacks' must be a list of keras callbacks.")
        # Checks on input values
        for k, v in dict(reg=reg, soften=soften, alpha=alpha).items():
            if k == "alpha":
                if not (isinstance(v, float) and 0.0 < v < 1.0):
                    raise ValueError(f"Argument '{k}' must be a float between 0 and 1")
            if not (isinstance(v, float) and v > 0.0):
                raise ValueError(f"Argument '{k}' must be a float > 0")
        if metrics is None:
            self.metrics = []
        else:
            self.metrics = metrics
        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks
        self.input_dim = input_dim
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.model = build_keras_piven(
            input_dim=input_dim,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self.model.compile(
            loss=piven_loss(eli=True, lambda_in=reg, soften=soften, alpha=alpha),
            optimizer=optimizer,
            metrics=self.metrics,
        )
        self.history = None
        self.optimizer = optimizer
        self.reg = reg
        self.soften = soften
        self.alpha = alpha

    def summary(self):
        self.model.summary()

    def fit(self, x: np.ndarray, y: np.ndarray):
        # Check y shape
        if len(y.shape) == 1:
            y = np.stack((y.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        elif len(y.shape) == 2:
            if y.shape[-1] == 1:
                y = np.stack((y, y), axis=1)
        else:
            raise ValueError(
                f"Incompatible number of dimensions found for y: ({y.shape})"
            )
        history = self.model.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            callbacks=self.callbacks,
            shuffle=True,
        )
        self.history = history.history
        return self

    def predict(
        self,
        x: np.ndarray,
        return_prediction_intervals=False,
        return_piven_estimator=True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.history is None:
            raise NotFittedError(
                "PIVEN has not yet been fitted. Run the 'fit()'"
                + " method before using 'predict()'"
            )
        # Predict on new data
        yhat = self.model.predict(x, verbose=0)
        # Upper / lower bounds
        y_upper_pred = yhat[:, 0]
        y_lower_pred = yhat[:, 1]
        y_value_pred = yhat[:, 2]
        # Point estimates
        if return_piven_estimator:
            y_out = y_value_pred * y_upper_pred + (1 - y_value_pred) * y_lower_pred
        else:
            y_out = 0.5 * y_upper_pred + 0.5 * y_lower_pred
        if return_prediction_intervals:
            return y_out.flatten(), y_lower_pred.flatten(), y_upper_pred.flatten()
        else:
            return y_out.flatten()

    def save(self, path: str):
        """Dump model to file"""
        plpath = pathlib.Path(path)
        if not plpath.is_dir():
            raise NotADirectoryError(f"Path '{path}' is not a directory")
        # Save configuration
        config = {
            "input_dim": self.input_dim,
            "optimizer": str(type(self.optimizer)),
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "verbose": self.verbose,
            "callbacks": None,
            "metrics": None,
            "reg": self.reg,
            "soften": self.soften,
            "alpha": self.alpha,
        }
        with (plpath / "config.json").open("w") as outfile:
            json.dump(config, outfile)
        # Save model
        tf.keras.models.save_model(self.model, (plpath / "piven_model.h5"))
        # Save history
        with (plpath / "history.json").open("w") as outfile:
            json.dump(self.history, outfile)
        return self

    @classmethod
    def load(cls, path: str):
        """Load model from file"""
        plpath = pathlib.Path(path)
        if not plpath.is_dir():
            raise NotADirectoryError(f"Path '{path}' is not a directory")
        # Check if all files exists
        files_in_dir = set([str(f).split("/")[-1] for f in plpath.iterdir()])
        for expected_file in ["config.json", "history.json", "piven_model.h5"]:
            if expected_file not in files_in_dir:
                raise FileNotFoundError(
                    f"Cannot load piven model from {path}. Missing model"
                    + f"file {expected_file}."
                )
        # Load config
        with (plpath / "config.json").open("r") as infile:
            config = json.load(infile)
        # Load model
        mod = tf.keras.models.load_model(plpath / "piven_model.h5")
        config["metrics"] = mod.metrics
        config["optimizer"] = mod.optimizer
        config["callbacks"] = []
        # Load history
        with (plpath / "history.json").open("r") as infile:
            history = json.load(infile)
        # Make class
        out_model = cls(**config)
        # Set model
        out_model.model = mod
        # Set history
        out_model.history = history
        return out_model
