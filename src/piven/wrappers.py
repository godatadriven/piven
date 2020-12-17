import types
import copy
from typing import Tuple, Union
import numpy as np
from piven.utils import piven_loss
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.models import Sequential

import tensorflow as tf
from piven.metrics import picp, mpiw
from piven.layers import Piven

# Dump custom metrics, loss and layers
# Need to do this when saving models.
tf.keras.utils.get_custom_objects().update(
    {"picp": picp, "mpiw": mpiw, "piven_loss": piven_loss, "Piven": Piven}
)


class PivenModelWrapper(KerasRegressor):
    def fit(self, x, y, **kwargs):
        """Fit the piven model"""
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
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(
            self.build_fn, types.MethodType
        ):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)

        history = self.model.fit(x, y, **fit_args)

        return history

    def predict(
        self, x, return_prediction_intervals=True, **kwargs
    ) -> Union[np.array, Tuple[np.array, np.array, np.array]]:
        """Predict method for a model with piven output layer"""
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        yhat = self.model.predict(x, **kwargs)
        # Upper / lower bounds
        y_upper_pred = yhat[:, 0]
        y_lower_pred = yhat[:, 1]
        y_value_pred = yhat[:, 2]
        if return_prediction_intervals:
            return y_value_pred, y_lower_pred, y_upper_pred
        else:
            return y_value_pred
