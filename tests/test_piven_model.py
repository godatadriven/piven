from piven.metrics.numpy import coverage, pi_distance
from sklearn.metrics import mean_squared_error
import numpy as np


class TestPivenModel:
    def test_fit_piven_model(self, mock_data, transformed_piven_regressor):
        # Get mock data
        x_train, x_valid, y_train, y_valid = mock_data
        transformed_piven_regressor.fit(
            x_train,
            y_train,
            model__epochs=500,
            model__validation_split=0.1,
            model__batch_size=64,
        )
        yhat, pi_low, pi_high = transformed_piven_regressor.predict(
            x_valid, return_prediction_intervals=True
        )
        xlinspace = np.linspace(-2, 2, x_train.shape[0])
        pred, lower, upper = transformed_piven_regressor.predict(
            xlinspace.reshape(-1, 1), return_prediction_intervals=True
        )
        # Plot if wanted in debug console
        # import matplotlib.pyplot as plt
        # plt.scatter(x_train, y_train)
        # plt.scatter(x_valid, y_valid)
        # plt.fill_between(xlinspace, lower, upper, color="blue", alpha=0.3)
        # plt.show()
        rmse = np.sqrt(mean_squared_error(y_valid, yhat))
        assert rmse < 2
        cov = coverage(y_valid, pi_low, pi_high)
        assert 0.85 < cov < 0.99
        # Distance between PIs
        pidist_lower = pi_distance(lower[xlinspace < -0.5], upper[xlinspace < -0.5])
        pidist_middle = pi_distance(
            lower[(xlinspace >= -0.5) & (xlinspace <= 0.5)],
            upper[(xlinspace >= -0.5) & (xlinspace <= 0.5)],
        )
        pidist_upper = pi_distance(lower[xlinspace > 0.5], upper[xlinspace > 0.5])
        assert pidist_upper > pidist_middle
        assert pidist_lower > pidist_middle
