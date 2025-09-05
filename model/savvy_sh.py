import numpy as np
from numpy.linalg import matrix_rank, inv, pinv
from sklearn.linear_model import RidgeCV, LinearRegression
from scipy.linalg import eigh

from model.linear_shrinkage import *
from model.multiplicative_shrinkage import *
from model.shrinkage_ridge import *
from model.slab_regression import *

class SavvySh:
    def __init__(self, model_class="Multiplicative", v=1, lambda_vals=None,
                 nlambda=100, folds=10, include_Sh=False, exclude=None):
        self.model_class = model_class
        self.v = v
        self.lambda_vals = lambda_vals
        self.nlambda = nlambda
        self.folds = folds
        self.include_Sh = include_Sh
        self.exclude = exclude

        self.ridge_results = None
        self.fitted_ = False
        self.coef_ = {}
        self.intercept_ = {}

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            The input features.
        y : array-like
            The target variable.
        
        Returns
        -------
        None
        """
        if X.shape[0] != len(y):
            raise ValueError("Incompatible shapes: X has {} rows, y has {} rows".format(X.shape[0], len(y)))
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input contains NaN values")
        if not self.model_class in ["Multiplicative", "Slab", "Linear", "ShrinkageRR"]:
            raise ValueError("Invalid model class: {}".format(self.model_class))
        if type(self.model_class) == list and len(self.model_class) > 1:
            print(f"model_class should be a single option. Using the first element: {self.model_class[0]}")
            self.model_class = self.model_class[0]
        if self.include_Sh and self.model_class != "Multiplicative":
            raise ValueError("Shrinkage estimator can only be included with Multiplicative model")

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        nobs, nvars = X.shape
        
        if self.exclude is not None:
            if all(isinstance(i, int) for i in self.exclude):
                X = np.delete(X, np.array(self.exclude) - 1, axis=1)
            else:
                raise ValueError("Exclude must be a list of column indices (1-based)")

        x_tilde = np.hstack([np.ones((nobs, 1)), X])
        full_rank = matrix_rank(X) == nvars

        # Ridge regression fallback if rank-deficient
        if self.model_class in ['Multiplicative', 'Slab']:
            sigma_square = None
            if not full_rank:
                print("Multicollinearity detected: switched to RR instead of unbiased OLS estimation")
                if self.lambda_vals is None:
                    self.lambda_vals = np.logspace(-6, 2, self.nlambda)
                ridge_cv = RidgeCV(alphas=self.lambda_vals, cv=self.folds)
                ridge_cv.fit(X, y)
                optimal_lambda = ridge_cv.alpha_
                est = np.hstack([ridge_cv.intercept_, ridge_cv.coef_])
                fitted_values = ridge_cv.predict(X)
                RSS = np.sum((y - fitted_values.reshape(-1, 1))**2)
                df_eff = np.sum(np.linalg.svd(X, compute_uv=False)**2 / 
                                (np.linalg.svd(X, compute_uv=False)**2 + optimal_lambda))
                sigma_square = (RSS / (nobs - df_eff))**2
                ridge_results = {"lambda_range": self.lambda_vals,
                                "ridge_coefficients": est}
                # print("Initial Fitted Succeed")
            else:
                lm = LinearRegression(fit_intercept=True)
                lm.fit(X, y)
                est = np.hstack([np.array([[lm.intercept_[0]]]), lm.coef_])
                sigma_square = np.var(y - lm.predict(X))
                optimal_lambda = 0
                # print("Initial Fitted Succeed")
            
            Sigma_lambda = x_tilde.T @ x_tilde + optimal_lambda * np.eye(x_tilde.shape[1])
            Sigma_lambda_inv = pinv(Sigma_lambda)

            # Multiplicative
            if self.model_class == "Multiplicative":
                est_St = stein_estimator(est, sigma_square, Sigma_lambda_inv)
                est_DSh = diagonal_shrinkage_estimator(est, sigma_square, Sigma_lambda_inv)
                self.coef_ = {"St": est_St[1:], "DSh": est_DSh[1:]}
                self.intercept_ = {"St": est_St[0], "DSh": est_DSh[0]}
                if self.include_Sh:
                    est_Sh = shrinkage_estimator(est, sigma_square, Sigma_lambda_inv)
                    self.coef_["Sh"] = est_Sh[1:]
                    self.intercept_["Sh"] = est_Sh[0]
                # print("Successfully fitted Multiplicative model")
            # Slab
            elif self.model_class == "Slab":
                est_SR = slab_estimator(self.v, est, sigma_square, Sigma_lambda, Sigma_lambda_inv)
                est_GSR = general_slab_regression(est, sigma_square, Sigma_lambda, Sigma_lambda_inv)
                self.coef_ = {"SR": est_SR[1:], "GSR": est_GSR[1:]}
                self.intercept_ = {"SR": est_SR[0], "GSR": est_GSR[0]}
                # print("Successfully fitted Slab model")
        # Linear
        elif self.model_class == "Linear":
            centered_X = X - X.mean(axis=0)
            centered_y = y - y.mean()
            sigma_square = None

            if not full_rank:
                print("Multicollinearity detected: switched to RR on centered data instead of unbiased OLS estimation")
                if self.lambda_vals is None:
                    self.lambda_vals = np.logspace(-6, 2, self.nlambda)
                ridge_cv = RidgeCV(alphas=self.lambda_vals, cv=self.folds)
                ridge_cv.fit(centered_X, centered_y)
                optimal_lambda = ridge_cv.alpha_
                est = ridge_cv.coef_.T
                fitted_values = ridge_cv.predict(centered_X)
                RSS = np.sum((centered_y - fitted_values.reshape(-1, 1))**2)
                df_eff = np.sum(np.linalg.svd(centered_X, compute_uv=False)**2 / 
                                (np.linalg.svd(centered_X, compute_uv=False)**2 + optimal_lambda))
                sigma_square = (RSS / (nobs - df_eff))**2
                ridge_results = {"lambda_range": self.lambda_vals,
                                "ridge_coefficients": est}
                # print("Initial Fitted Succeed")
            else:
                lm = LinearRegression(fit_intercept=False)
                lm.fit(centered_X, centered_y)
                est = lm.coef_.T
                sigma_square = np.var(centered_y - lm.predict(centered_X))
                optimal_lambda = 0
                # print("Initial Fitted Succeed")

            Sigma_lambda = centered_X.T @ centered_X + optimal_lambda * np.eye(centered_X.shape[1])
            Sigma_lambda_inv = pinv(Sigma_lambda)
            Sigma_tilde = np.diag(np.diag(Sigma_lambda))

            est_LSh = linear_shrinkage(est, sigma_square, Sigma_lambda, Sigma_lambda_inv, Sigma_tilde)
            self.coef_ = {"LSh": est_LSh}
            self.intercept_ = {"LSh": 0}
            # print("Successfully fitted Linear model")
        # Shrinkage Ridge Regression
        elif self.model_class == "ShrinkageRR":
            Sigma_lambda = x_tilde.T @ x_tilde

            est_SRR = shrinkage_ridge_regression(x_tilde, y, Sigma_lambda)
            self.coef_ = {"SRR": est_SRR[1:]}
            self.intercept_ = {"SRR": est_SRR[0]}
            # print("Successfully fitted Shrinkage Ridge model")

        if not full_rank and self.model_class != "ShrinkageRR":
            self.ridge_results = ridge_results

        self.fitted_ = True

    def predict(self, X, estimator=None) -> dict | np.ndarray:
        """
        Predict response values using the fitted model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input features.
        estimator : str, optional
            Which estimator to use for prediction.
            For Multiplicative: "St", "DSh", "Sh"
            For Slab: "SR", "GSR"
            For Linear: "LSh"
            For ShrinkageRR: "SRR"
            If None, will return predictions for all available estimators.

        Returns
        -------
        predictions : dict or np.ndarray
            If estimator is None, returns dict of predictions for all estimators.
            If estimator is specified, returns np.ndarray of predictions.
        """
        if not self.fitted_:
            raise RuntimeError("You must fit the model before prediction.")

        X = np.asarray(X)
        n_samples = X.shape[0]

        X_tilde = np.hstack([np.ones((n_samples, 1)), X])

        if estimator is not None:
            if estimator not in self.coef_:
                raise ValueError(f"Estimator {estimator} not available in this model.")
            if estimator == "LSh":
                return X @ self.coef_[estimator].flatten()
            else:
                return X_tilde @ np.concatenate(([self.intercept_[estimator]], self.coef_[estimator])).flatten()

        preds = {}
        for key, coef in self.coef_.items():
            if key == "LSh":
                preds[key] = X @ coef.flatten()
            else:
                preds[key] = X_tilde @ np.concatenate(([self.intercept_[key]], coef)).flatten()

        return preds