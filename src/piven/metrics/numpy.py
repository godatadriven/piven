import numpy as np


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
