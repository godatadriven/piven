from typing import Tuple
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from piven.experiments.base import PivenExperiment
from piven.regressors import build_keras_piven
from piven.scikit_learn.compose import PivenTransformedTargetRegressor
from piven.scikit_learn.wrappers import PivenKerasRegressor
from piven.metrics.tensorflow import picp, mpiw
from piven.loss import piven_loss
from piven.metrics.numpy import coverage, pi_width, piven_loss as piven_loss_numpy


def check_model_params(
    hidden_units: Tuple[int, ...],
    dropout_rate: Tuple[float, ...],
    pi_init_low: float,
    pi_init_high: float,
    lambda_: float,
    lr: float,
) -> None:
    should_be_float = [*dropout_rate] + [pi_init_low, pi_init_high, lambda_, lr]
    should_be_float_argnames = [
        f"dropout_value_{idx + 1}" for idx in range(len(dropout_rate))
    ] + ["pi_init_low", "pi_init_high", "lambda_", "learning_rate"]
    for argname, argval in zip(should_be_float_argnames, should_be_float):
        if argname != "pi_init_low":
            if not isinstance(argval, float) or argval < 0.0:
                raise ValueError(f"Argument {argname} should be a float > 0.")
        else:
            if not isinstance(argval, float):
                raise ValueError(f"Argument {argname} should be a float.")
    for argval in hidden_units:
        if not isinstance(argval, int) or argval < 1:
            raise ValueError("All hidden units should be integers > 0.")
    if not len(hidden_units) == len(dropout_rate):
        raise ValueError(
            "Number of hidden units"
            + f" in each layer {len(hidden_units)} and dropout rate"
            + f" in each layer {len(dropout_rate)} must be equal."
        )
    if pi_init_low > pi_init_high:
        raise ValueError(
            f"Value of lower pi {pi_init_low} is larger than the"
            + f" value of the upper pi {pi_init_high}."
        )
    return None


# Make build function for the model wrapper
def piven_model(input_dim, dense_units, dropout_rate, lambda_=10.0, lr=0.0001):
    model = build_keras_piven(
        input_dim=input_dim,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        activation="relu",
        bias_init_low=-3,
        bias_init_high=3,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=piven_loss(lambda_, 160.0, 0.05),
        metrics=[picp, mpiw],
    )
    return model


class PivenMlpExperiment(PivenExperiment):
    def build_model(self):
        # All build params are passed to init and should be checked here
        check_model_params(**self.params)
        model = PivenKerasRegressor(build_fn=build_keras_piven, **self.params)
        pipeline = Pipeline([("preprocess", StandardScaler()), ("model", model)])
        # Finally, normalize the output target
        self.model = PivenTransformedTargetRegressor(
            regressor=pipeline, transformer=StandardScaler()
        )
        return self.model

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pi_low: np.ndarray,
        y_pi_high: np.ndarray,
    ):
        # Compute coverage, pi width and loss
        return {
            "loss": piven_loss_numpy(
                y_true,
                y_pred,
                y_pi_low,
                y_pi_high,
                self.params.get("lambda_"),
                160.0,
                0.05,
            ),
            "coverage": coverage(y_true, y_pi_low, y_pi_high),
            "pi_width": pi_width(y_pi_low, y_pi_high),
        }
