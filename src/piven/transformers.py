from typing import Union
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.compose import TransformedTargetRegressor as _TTR
from sklearn.pipeline import Pipeline
from piven.wrappers import PivenModelWrapper


class PivenTransformedTargetRegressor(_TTR):
    @_deprecate_positional_args
    def __init__(
        self,
        regressor: Union[PivenModelWrapper, Pipeline] = None,
        *,
        transformer=None,
        func=None,
        inverse_func=None,
        check_inverse=True
    ):
        super().__init__()
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse

    def predict(self, x, return_prediction_intervals=None):
        """
        Overwrite the base class predict method to allow
        for additional parameters to be passed to predict method
        """
        check_is_fitted(self)
        if return_prediction_intervals is None:
            return_prediction_intervals = False
        if return_prediction_intervals:
            pred, pi_lower, pi_higher = self.regressor_.predict(
                x, return_prediction_intervals=return_prediction_intervals
            )
        else:
            pred = self.regressor_.predict(
                x, return_prediction_intervals=return_prediction_intervals
            )
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
            if return_prediction_intervals:
                pi_lower_trans = self.transformer_.inverse_transform(
                    pi_lower.reshape(-1, 1)
                )
                pi_higher_trans = self.transformer_.inverse_transform(
                    pi_higher.reshape(-1, 1)
                )
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if (
            self._training_dim == 1
            and pred_trans.ndim == 2
            and pred_trans.shape[1] == 1
        ):
            pred_trans = pred_trans.squeeze(axis=1)

        if return_prediction_intervals:
            return (
                pred_trans.flatten(),
                pi_lower_trans.flatten(),
                pi_higher_trans.flatten(),
            )
        else:
            return pred_trans.flatten()
