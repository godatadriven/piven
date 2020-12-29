from typing import Callable
import tensorflow as tf


def piven_loss(
    lambda_in: float = 15.0, soften: float = 160.0, alpha: float = 0.05
) -> Callable:
    """
    Piven loss function

    :param lambda_in:
    :param soften:
    :param alpha:
    :return: tf loss function for Piven models.

    Taken from:

        Simhayev, Eli, Gilad Katz, and Lior Rokach. "PIVEN: A Deep
        Neural Network for Prediction Intervals with Specific Value
        Prediction." arXiv preprint arXiv:2006.05139 (2020).
    """
    # define loss fn
    def piven_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        y_u = y_pred[:, 0]
        y_l = y_pred[:, 1]
        y_t = y_true[:, 0]
        y_v = y_pred[:, 2]
        n_ = tf.cast(tf.size(y_t), tf.float32)  # batch size
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)
        # soft uses sigmoid
        k_soft = tf.multiply(
            tf.sigmoid((y_u - y_t) * soften), tf.sigmoid((y_t - y_l) * soften)
        )
        # hard uses sign step function
        k_hard = tf.multiply(
            tf.maximum(0.0, tf.sign(y_u - y_t)), tf.maximum(0.0, tf.sign(y_t - y_l))
        )
        mpiw_capt = tf.divide(
            tf.reduce_sum(tf.abs(y_u - y_l) * k_hard), tf.reduce_sum(k_hard) + 0.001
        )
        picp_soft = tf.reduce_mean(k_soft)
        qd_rhs_soft = (
            lambda_
            * tf.sqrt(n_)
            * tf.square(tf.maximum(0.0, (1.0 - alpha_) - picp_soft))
        )
        piven_loss_ = mpiw_capt + qd_rhs_soft  # final qd loss form
        y_eli = y_v * y_u + (1 - y_v) * y_l
        piven_loss_ += tf.losses.mean_squared_error(y_t, y_eli)
        return piven_loss_

    return piven_loss
