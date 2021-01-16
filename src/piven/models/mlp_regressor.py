from typing import Tuple, Union
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from piven.models.base import PivenBaseModel
from piven.regressors import build_keras_piven
from piven.scikit_learn.compose import PivenTransformedTargetRegressor
from piven.scikit_learn.wrappers import PivenKerasRegressor
from piven.metrics.tensorflow import picp, mpiw
from piven.loss import piven_loss


def check_model_params(
    input_dim: int,
    dense_units: Tuple[int, ...],
    dropout_rate: Tuple[float, ...],
    bias_init_low: float,
    bias_init_high: float,
    lambda_: float,
    lr: float,
) -> None:
    should_be_float = [*dropout_rate] + [bias_init_low, bias_init_high, lambda_, lr]
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
    for argval in dense_units:
        if not isinstance(argval, int) or argval < 1:
            raise ValueError("All dense units should be integers > 0.")
    if not len(dense_units) == len(dropout_rate):
        raise ValueError(
            "Number of dense units"
            + f" in each layer {len(dense_units)} and dropout rate"
            + f" in each layer {len(dropout_rate)} must be equal."
        )
    if input_dim < 1:
        raise ValueError("Input dimension cannot be < 1.")
    if bias_init_low > bias_init_high:
        raise ValueError(
            f"Value of lower pi {bias_init_low} is larger than the"
            + f" value of the upper pi {bias_init_high}."
        )
    return None


# Make build function for the model wrapper
def piven_mlp_model(
    input_dim, dense_units, dropout_rate, lambda_, bias_init_low, bias_init_high, lr
):
    model = build_keras_piven(
        input_dim=input_dim,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        activation="relu",
        bias_init_low=bias_init_low,
        bias_init_high=bias_init_high,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=piven_loss(lambda_, 160.0, 0.05),
        metrics=[picp, mpiw],
    )
    return model


class PivenMlpModel(PivenBaseModel):
    def build(
        self,
        build_fn=piven_mlp_model,
        preprocess: Union[None, Pipeline, TransformerMixin] = None,
    ):
        # All build params are passed to init and should be checked here
        check_model_params(**self.params)
        model = PivenKerasRegressor(build_fn=build_fn, **self.params)
        if preprocess is None:
            pipeline = Pipeline([("model", model)])
        else:
            pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
        # Finally, normalize the output target
        self.model = PivenTransformedTargetRegressor(
            regressor=pipeline, transformer=StandardScaler()
        )
        return self
