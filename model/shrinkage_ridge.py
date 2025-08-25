import numpy as np
from scipy.optimize import minimize_scalar

def h_function(rho, nobs, p_plus_1, yty, lambdas, V_k, V_star):
    """
    Helper function to compute H(rho) for SRR model.

    Parameters
    ----------
    rho: float
        Shrinkage parameter
    nobs: int
        Number of observations
    p_plus_1: int
        Number of predictors plus one (for intercept)
    yty: float
        y^T y
    lambdas: np.ndarray
        Eigenvalues of the regularized covariance matrix
    V_k: np.ndarray
        Projected response variances
    V_star: float
        Mean of the eigenvalues

    Returns
    -------
    float
        Value of H(rho)
    """
    denom = (1 - rho) * lambdas + rho * V_star
    denom2 = denom ** 2
    denom3 = denom ** 3
    term1 = np.sum(lambdas / denom2)
    term2 = np.sum(V_k / denom)
    term3 = np.sum(lambdas * V_k / denom2)
    term4 = np.sum(V_k * rho**2 * (lambdas - V_star)**2 / denom3)
    H_rho = (1 / (nobs - p_plus_1)) * (yty - 2 * term2 + term3) * term1 + term4
    return H_rho

def find_rho_star(nobs, p_plus_1, yty, lambdas, V_k, V_star):
    """
    Find optimal rho that minimizes H(rho) in [0, 1].

    Parameters
    ----------
    nobs: int
        Number of observations
    p_plus_1: int
        Number of predictors plus one (for intercept)
    yty: float
        y^T y
    lambdas: np.ndarray
        Eigenvalues of the regularized covariance matrix
    V_k: np.ndarray
        Projected response variances
    V_star: float
        Mean of the eigenvalues

    Returns
    -------
    float
        Optimal rho
    """
    res = minimize_scalar(lambda rho: h_function(rho, nobs, p_plus_1, yty, lambdas, V_k, V_star),
                          bounds=(0, 1), method='bounded')
    return res.x

def shrinkage_ridge_regression(x_tilde, y, Sigma_lambda):
    """
    Compute Shrinkage Ridge Regression estimator.

    Parameters
    ----------
    x_tilde: np.ndarray
        Design matrix including intercept
    y: np.ndarray
        Response vector
    Sigma_lambda: np.ndarray
        Regularized covariance matrix (X^T X + lambda*I)

    Returns
    -------
    np.ndarray
        Shrinkage Ridge Regression estimator
    """
    nobs, p_plus_1 = x_tilde.shape
    # Eigen decomposition
    lambdas, mu = np.linalg.eigh(Sigma_lambda)  # symmetric matrix
    V_star = np.mean(lambdas)
    yty = float(y.T @ y)
    proj = (y.T @ (x_tilde @ mu)).flatten()
    V_k = proj**2
    rho_star = find_rho_star(nobs, p_plus_1, yty, lambdas, V_k, V_star)
    diag_entries = 1 / ((1 - rho_star) * lambdas + rho_star * V_star)
    beta_SRR = mu @ np.diag(diag_entries) @ mu.T
    est_SRR = beta_SRR @ (x_tilde.T @ y)
    return est_SRR.flatten()
