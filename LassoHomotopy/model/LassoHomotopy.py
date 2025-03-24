import numpy as np

class LassoHomotopyModel():
    def __init__(self, **kwargs):
        # Allow optional tuning parameters; use defaults if not provided.
        self.tol = kwargs.get('tol', 1e-4)
        self.max_iter = kwargs.get('max_iter', 1000)
        self.coef_ = None

    def fit(self, X, y):
        """
        Fits the LASSO regression model using a simplified Homotopy (LARS-like) method.
        Returns a LassoHomotopyResults object containing the fitted coefficients.
        """
        n, p = X.shape
        beta = np.zeros(p)
        residual = y.copy()
        # Starting lambda: maximum absolute correlation
        lambda_val = np.max(np.abs(X.T @ y))
        iter_count = 0
        active_set = []

        while lambda_val > self.tol and iter_count < self.max_iter:
            # Compute correlations between predictors and the residual
            corr = X.T @ residual
            # Identify the variable with the maximum absolute correlation
            j = np.argmax(np.abs(corr))
            if j not in active_set:
                active_set.append(j)

            # Solve least squares on the active set
            X_active = X[:, active_set]
            beta_active, _, _, _ = np.linalg.lstsq(X_active, y, rcond=None)

            beta_new = beta.copy()
            for idx, var in enumerate(active_set):
                beta_new[var] = beta_active[idx]

            # Update residual and lambda value
            residual_new = y - X @ beta_new
            lambda_val = np.max(np.abs(X.T @ residual_new))
            beta = beta_new
            residual = residual_new
            iter_count += 1

        self.coef_ = beta
        return LassoHomotopyResults(self.coef_)


class LassoHomotopyResults():
    def __init__(self, coef):
        """
        Stores the fitted coefficients.
        """
        self.coef = coef

    def predict(self, X):
        """
        Predicts responses for input matrix X using the fitted coefficients.
        """
        return np.dot(X, self.coef)
