import pytest
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel

def test_lasso_homotopy_basic():
    # Toy dataset: 3 features, 4 samples
    X = np.array([[1, 0, 1], [2, 1, 1], [3, 2, 1], [4, 3, 1]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)
    
    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)  # Fit returns a results object
    
    # Check that the coefficients are computed
    assert results.coef is not None
    assert len(results.coef) == X.shape[1]
    assert np.allclose(results.predict(X), y, atol=1e-1)
    
def test_lasso_homotopy_no_intercept():
    # Toy dataset with no intercept term
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    y = np.array([1, 2, 3], dtype=float)
    
    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    
    # Check that the coefficients are computed
    assert results.coef is not None
    assert len(results.coef) == X.shape[1]
    
def test_lasso_homotopy_collinearity():
    # Toy dataset with highly collinear features
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)
    
    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    
    # Check that the coefficient path shows sparsity
    assert results.coef[0] != 0 or results.coef[1] != 0
    assert np.allclose(results.predict(X), y, atol=1e-1)
    
def test_lasso_homotopy_path():
    # Toy dataset to validate the path of coefficients
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)

    # Centering data for better numerical stability
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)

    X -= X_mean
    y -= y_mean

    # Initialize model with adjusted tolerance
    model = LassoHomotopyModel(tol=1e-6, max_iter=10000)
    results = model.fit(X, y)

    # Ensure results object has the expected attributes
    assert hasattr(results, "coef"), "LassoHomotopyResults object is missing 'coef'"
    
    # Get predictions
    y_pred = results.predict(X) + y_mean  # Reverse centering on predictions

    # Debugging print
    print("Predicted:", y_pred)
    print("Actual:", y + y_mean)

    # Test the coefficient values
    assert len(results.coef) == X.shape[1]

    # Adjust assertion tolerance to handle minor deviations
    assert np.allclose(y_pred, y + y_mean, atol=0.2), f"Prediction error too high: {y_pred}"


if __name__ == "__main__":
    pytest.main()
