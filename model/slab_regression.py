import numpy as np

def slab_estimator(v, est, sigma_square, Sigma_lambda, Sigma_lambda_inv):
    """
    Slab estimator

    Parameters
    ----------
    v: float
        Value to construct vector u
    est: np.ndarray
        Coefficient vector
    sigma_square: float
        Variance of noise
    Sigma_lambda: np.ndarray
        Regularized covariance matrix
    Sigma_lambda_inv: np.ndarray
        Inverse of Sigma_lambda
    
    Returns
    -------
    np.ndarray
        Slab estimator
    """
    p = len(est.T)
    u = np.full((p, 1), v)
    
    a0_u = np.sum(u**2)
    a1_u = float(u.T @ np.linalg.matrix_power(Sigma_lambda_inv, 1) @ u)
    a2_u = float(u.T @ np.linalg.matrix_power(Sigma_lambda_inv, 2) @ u)
    a3_u = float(u.T @ np.linalg.matrix_power(Sigma_lambda_inv, 3) @ u)
    
    delta_u = sigma_square * (a0_u * a3_u - a1_u * a2_u) + a3_u * (float(u.T @ est.T))**2
    mu_star = (sigma_square * a2_u) / delta_u
    J = np.ones((p, p))
    
    if delta_u > 0:
        scalar_value = mu_star / (1 + mu_star * a1_u)
        est_SR = (np.eye(p) - scalar_value * Sigma_lambda_inv @ J) @ est.T
    else:
        est_SR = (np.eye(p) - Sigma_lambda_inv @ J) @ est.T
    return est_SR.flatten()

def general_slab_regression(est, sigma_square, Sigma_lambda, Sigma_lambda_inv=None):
    """
    General Slab Regression (GSR) estimator

    Parameters
    ----------
    est: np.ndarray
        Coefficient vector
    sigma_square: float
        Variance of noise
    Sigma_lambda: np.ndarray
        Regularized covariance matrix
    Sigma_lambda_inv: np.ndarray, optional
        Inverse of Sigma_lambda

    Returns
    -------
    np.ndarray
        General Slab Regression (GSR) estimator
    """
    eigenvalues, U = np.linalg.eigh(Sigma_lambda)  # symmetric
    beta_proj = U @ est.T
    mu_l_star = sigma_square / (beta_proj**2)
    scalar_coeff = (mu_l_star / eigenvalues) / (1 + (mu_l_star / eigenvalues))
    adjustment_matrix = np.eye(len(est.T)) - U @ np.diag(scalar_coeff) @ U.T
    return (adjustment_matrix @ est.T).flatten()
