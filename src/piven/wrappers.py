import types
import copy
from typing import Tuple, Union
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.python.keras.models import Sequential


class PivenModelWrapper(KerasRegressor):
    def __init__(self, build_fn=None, **sk_params):
        super().__init__(build_fn, **sk_params)
        self.history = None

    def fit(self, x, y, **kwargs):
        """Fit the piven model"""
        # Check y shape and stack model vertically. (upper and lower bound).
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
        self.history = history.history
        return self.model

    def predict(
        self, x, return_prediction_intervals=True, return_pi_weight=False, **kwargs
    ) -> Union[np.array, Tuple[np.array, np.array, np.array]]:
        """
        Predict method for a model with piven output layer
        :param x:
        :param return_pi_weight:
        :param return_prediction_intervals:
        :return:
        """
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        yhat = self.model.predict(x, **kwargs)
        # Upper / lower bounds
        y_upper_pred = yhat[:, 0]
        y_lower_pred = yhat[:, 1]
        # Point prediction
        # This is a weighted sum of the sigmoid * upper/lower bounds
        y_value_pred = yhat[:, 2]
        # If want the sigmoid weights
        if return_pi_weight:
            y_out = y_value_pred.copy()
        else:
            y_out = y_value_pred * y_upper_pred + (1 - y_value_pred) * y_lower_pred
        if return_prediction_intervals:
            return y_out.flatten(), y_lower_pred.flatten(), y_upper_pred.flatten()
        else:
            return y_out.flatten()
