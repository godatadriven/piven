import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.special import expit


def coverage(y_true: np.ndarray, pi_lower: np.ndarray, pi_higher: np.ndarray) -> float:
    """Compute the coverage of a PIVEN estimator"""
    covered_lower = pi_lower < y_true
    covered_higher = pi_higher > y_true
    return float(np.mean(covered_lower * covered_higher))


def pi_width(pi_lower: np.ndarray, pi_higher: np.ndarray) -> float:
    """Compute the average distance between the upper and lower bounds"""
    return float(np.mean(pi_higher - pi_lower))


def piven_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_pi_low: np.ndarray,
    y_pred_pi_high: np.ndarray,
    lambda_in: float,
    soften: float,
    alpha: float,
) -> float:
    """
    Compute piven loss for predictions.
    """
    n = y_true.shape[0]
    k_soft = _sigmoid((y_pred_pi_high - y_true) * soften) * _sigmoid(
        (y_true - y_pred_pi_low) * soften
    )
    # 1 if obs between lower & upper PI, else 0
    k_hard = np.maximum(0.0, np.sign(y_pred_pi_high - y_true)) * np.maximum(
        0.0, np.sign(y_true - y_pred_pi_low)
    )
    # Average of points between lower & upper PI
    mpiw_capt = (
        np.divide(
            np.sum(np.abs(y_pred_pi_high - y_pred_pi_low) * k_hard),
            np.sum(k_hard) + 0.001,
        ),
    )
    # Coverage
    picp_soft = np.mean(k_soft)
    # Interior point method --> lambda controls relative importantce of width v. coverage.
    qd_rhs_soft = (
        lambda_in * np.sqrt(n) * np.square(np.maximum(0.0, (1.0 - alpha) - picp_soft))
    )
    piven_loss_ = mpiw_capt + qd_rhs_soft
    piven_loss_ += mean_squared_error(y_true, y_pred)
    return float(piven_loss_)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)
