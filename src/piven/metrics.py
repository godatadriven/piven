import tensorflow as tf
import numpy as np


# NB: need y_true in mpiw even if not using to be able to pass to keras.
#      this is a required argument.
def mpiw(y_true, y_pred):
    """Compute the point estimate as the middle of the upper & lower CI"""
    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]
    return tf.reduce_mean(y_u_pred - y_l_pred)


def picp(y_true, y_pred):
    """Prediction Interval Coverage Percentage of an estimator"""
    y_true = y_true[:, 0]
    y_u_pred = y_pred[:, 0]
    y_l_pred = y_pred[:, 1]
    k_u = tf.cast(y_u_pred > y_true, tf.float32)
    k_l = tf.cast(y_l_pred < y_true, tf.float32)
    return tf.reduce_mean(k_l * k_u)


def coverage(
    y_true: np.ndarray, pi_lower: np.ndarray, pi_higher: np.ndarray
) -> np.ndarray:
    """Compute the coverage of a PIVEN estimator"""
    covered_lower = pi_lower < y_true
    covered_higher = pi_higher > y_true
    return np.mean(covered_lower * covered_higher)


def pi_distance(pi_lower: np.ndarray, pi_higher: np.ndarray) -> np.ndarray:
    """Compute the average distance between the upper and lower bounds"""
    return np.mean(pi_higher - pi_lower)
