import numpy as np

def linear_shrinkage(est, sigma_square, Sigma_lambda, Sigma_lambda_inv, Sigma_tilde):
    """
    Linear Shrinkage Estimator (LSh)

    Parameters
    ----------
    est : np.ndarray
        Initial coefficient estimate vector (shape (p,))
    sigma_square : float
        Variance of noise
    Sigma_lambda : np.ndarray
        Regularized covariance matrix (p x p)
    Sigma_lambda_inv : np.ndarray
        Inverse of Sigma_lambda (p x p)
    Sigma_tilde : np.ndarray
        Shrinkage target covariance matrix (p x p)

    Returns
    -------
    np.ndarray
        Shrinked coefficient vector (shape (p,))
    """
    p = len(est)
    I_p = np.eye(p)

    # Inverse of Sigma_tilde (regularize if singular)
    Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
    
    # t1 and t2 (trace terms)
    t1 = sigma_square * np.trace(Sigma_tilde_inv)
    t2 = sigma_square * np.trace(Sigma_lambda_inv)

    # difference matrix
    diff_matrix = Sigma_tilde_inv @ Sigma_lambda - I_p

    # quadratic form
    t3 = float(est.T @ (diff_matrix @ diff_matrix) @ est)

    # shrinkage intensity
    rho_star = (t2 - t1) / (t2 - t1 + t3) if (t2 - t1 + t3) != 0 else 0.0

    # shrinkage matrix
    Sigma_rho_star = rho_star * (Sigma_tilde_inv @ Sigma_lambda) + (1 - rho_star) * I_p

    return (Sigma_rho_star @ est).flatten()
