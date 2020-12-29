import tensorflow as tf

# NB: need y_true in mpiw even if not using to be able to pass to keras.
#      this is a required argument.


def mpiw(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute the point estimate as the middle of the upper & lower CI

    Taken from:

        Simhayev, Eli, Gilad Katz, and Lior Rokach. "PIVEN: A Deep
        Neural Network for Prediction Intervals with Specific Value
        Prediction." arXiv preprint arXiv:2006.05139 (2020).
    """
    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]
    return tf.reduce_mean(y_u_pred - y_l_pred)


def picp(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Prediction Interval Coverage Percentage of an estimator

    Taken from:

        Simhayev, Eli, Gilad Katz, and Lior Rokach. "PIVEN: A Deep
        Neural Network for Prediction Intervals with Specific Value
        Prediction." arXiv preprint arXiv:2006.05139 (2020).
    """
    y_true = y_true[:, 0]
    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]
    k_u = tf.cast(y_u_pred > y_true, tf.float32)
    k_l = tf.cast(y_l_pred < y_true, tf.float32)
    return tf.reduce_mean(k_l * k_u)
