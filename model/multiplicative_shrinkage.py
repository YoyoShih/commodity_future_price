import numpy as np
from scipy.linalg import solve_sylvester

def stein_estimator(est, sigma_square, Sigma_lambda_inv):
    """
    Stein estimator (St) using multiplicative shrinkage.

    Parameters
    ----------
    est: np.ndarray
        coefficient vector
    sigma_square: float
        variance of noise
    Sigma_lambda_inv: np.ndarray
        inverse of regularized covariance matrix

    Returns
    -------
    np.ndarray
        Shrinkage estimator
    """
    M0 = sigma_square * np.trace(Sigma_lambda_inv)
    a_star = np.sum(est**2) / (np.sum(est**2) + M0)
    return (a_star * est).flatten()

def diagonal_shrinkage_estimator(est, sigma_square, Sigma_lambda_inv):
    """
    Diagonal Shrinkage (DSh) estimator.

    Parameters
    ----------
    est: np.ndarray
        coefficient vector
    sigma_square: float
        variance of noise
    Sigma_lambda_inv: np.ndarray
        inverse of regularized covariance matrix

    Returns
    -------
    np.ndarray
        Shrinkage estimator
    """
    b_star = (est.ravel()**2) / (est.ravel()**2 + sigma_square * np.diag(Sigma_lambda_inv))
    return (b_star * est.ravel()).flatten()

def shrinkage_estimator(est, sigma_square, Sigma_lambda_inv):
    """
    Shrinkage (Sh) estimator using Sylvester equation.

    Parameters
    ----------
    est: np.ndarray
        coefficient vector
    sigma_square: float
        variance of noise
    Sigma_lambda_inv: np.ndarray
        inverse of regularized covariance matrix

    Returns
    -------
    np.ndarray
        Shrinkage estimator
    """
    B = np.outer(est, est)
    # Solve Sylvester equation: A X + X B = C
    # Here: A = Sigma_lambda_inv, B = B, C = B
    C_star = solve_sylvester(Sigma_lambda_inv, B, B)
    return (C_star @ est.T).flatten()
