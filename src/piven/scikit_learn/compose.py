from sklearn.utils.validation import check_is_fitted
from sklearn.compose import TransformedTargetRegressor as _TTR


class PivenTransformedTargetRegressor(_TTR):
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
