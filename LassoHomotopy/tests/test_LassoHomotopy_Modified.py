import csv
import numpy as np
import os
import pytest
from model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    # Load data from small_test_new.csv in the tests folder
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "small_test.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features (columns starting with 'x') and target (column 'y')
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    # Fit model
    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    preds = results.predict(X)
    
    # Instead of expecting a constant value, check that the predictions are reasonably close to the targets.
    mse = np.mean((preds - y)**2)
    assert mse < 12.0, f"Mean squared error {mse} is too high"


def test_collinear_data():
    """
    Test the LASSO Homotopy model with collinear data.
    We load data from 'collinear_data.csv' which contains features named X_0, X_1, ... X_10 and a target column.
    This test checks both the prediction quality and that the model produces a sparse solution.
    """
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "collinear_data.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # Extract features (columns starting with 'X_') and target (column 'target')
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('X_')] for datum in data])
    y = np.array([float(datum["target"]) for datum in data])
    
    # Fit model
    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    preds = results.predict(X)
    
    mse = np.mean((preds - y)**2)
    assert mse < 5.0, f"Mean squared error {mse} is too high for collinear data"
    
    # Check sparsity: count the number of coefficients close to zero.
    # Adjust the threshold as needed for your implementation.
    sparse_coef_count = np.sum(np.abs(results.coef) < 1e-2)
    assert sparse_coef_count > 0, "Expected at least one coefficient to be near zero for collinear data"


def test_new_test_data():
    """
    Test the model on the small_test_old.csv dataset.
    This CSV is expected to have columns named 'x_0', 'x_1', 'x_2' and 'y'.
    The test checks that the model predictions yield a low mean squared error.
    """
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "new_test_data.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # Extract features (columns starting with 'x') and target (column 'y')
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    
    # Fit model
    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    preds = results.predict(X)
    
    mse = np.mean((preds - y)**2)
    assert mse < 1.0, f"Mean squared error {mse} is too high for small_test_old data"
