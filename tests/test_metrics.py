import pytest
import numpy as np
import tensorflow as tf
from piven.metrics.numpy import piven_loss, coverage, pi_width
from piven.loss import piven_loss as piven_loss_tf


@pytest.fixture(scope="class")
def mock_data(request):
    np.random.seed(6628673)
    y_true = np.random.normal(0, 1, 100)
    y_pi_low = (y_true.copy() + np.random.uniform(-1, 0, 100)).reshape(-1, 1)
    y_pi_high = (y_true.copy() + np.random.uniform(0, 1, 100)).reshape(-1, 1)
    y_weight = np.repeat(0.5, 100).reshape(-1, 1)
    # Add some noise to y_true
    y_true += np.random.normal(0, 0.2, 100)
    y_pred = y_weight * y_pi_high + (1 - y_weight) * y_pi_low
    request.cls.y_true = y_true.reshape(-1, 1)
    request.cls.y_pred = y_pred
    request.cls.y_pi_low = y_pi_low
    request.cls.y_pi_high = y_pi_high
    request.cls.y_weight = y_weight


@pytest.mark.usefixtures("mock_data")
class TestMetrics:
    def test_piven_loss(self):
        loss = piven_loss(
            self.y_true, self.y_pred, self.y_pi_low, self.y_pi_high, 15.0, 160.0, 0.05
        )
        assert np.round(loss, 3) == 3.093

    def test_piven_loss_np_equals_tf(self):
        loss_np = piven_loss(
            self.y_true, self.y_pred, self.y_pi_low, self.y_pi_high, 15.0, 160.0, 0.05
        )
        y_pred_prep = tf.convert_to_tensor(
            np.concatenate(
                [self.y_pi_high.copy(), self.y_pi_low.copy(), self.y_weight.copy()],
                axis=1,
            ),
            dtype=tf.float32,
        )
        y_true_prep = tf.convert_to_tensor(
            np.concatenate(
                [self.y_true.copy().reshape(-1, 1), self.y_true.copy().reshape(-1, 1)],
                axis=1,
            ),
            dtype=tf.float32,
        )
        loss_tf = piven_loss_tf(15.0, 160.0, 0.05)(y_true_prep, y_pred_prep).numpy()
        np.testing.assert_almost_equal(loss_np, loss_tf, decimal=4)

    def test_coverage(self):
        cov = coverage(self.y_true, self.y_pi_low, self.y_pi_high)
        assert cov == 0.84

    def test_pi_width(self):
        piw = pi_width(self.y_pi_low, self.y_pi_high)
        assert np.round(piw, 3) == 0.943
