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
        # Predict
        pred = self.regressor_.predict(
            x, return_prediction_intervals=return_prediction_intervals
        )
        if return_prediction_intervals:
            return (
                self.transformer_.inverse_transform(pred[0].reshape(-1, 1)).flatten(),
                self.transformer_.inverse_transform(pred[1].reshape(-1, 1)).flatten(),
                self.transformer_.inverse_transform(pred[2].reshape(-1, 1)).flatten(),
            )
        else:
            return self.transformer_.inverse_transform(pred.reshape(-1, 1)).flatten()
