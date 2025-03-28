import csv
import numpy as np
import os
import pytest
from model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "small_test.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    preds = results.predict(X)

    mse = np.mean((preds - y)**2)
    r2_score = 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(mse)

    print(f"\n small_test.csv")
    print(f"MSE for small_test.csv: {mse:.4f}")
    print(f"R-squared for small_test.csv: {r2_score:.4f}")
    print(f"RMSE for small_test.csv: {rmse:.4f}")

    assert mse < 12.0, f"Mean squared error {mse} is too high"

def test_collinear_data():
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "collinear_data.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('X_')] for datum in data])
    y = np.array([float(datum["target"]) for datum in data])

    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    preds = results.predict(X)

    mse = np.mean((preds - y)**2)
    r2_score = 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(mse)

    print(f"\n collinear_data.csv")
    print(f"MSE for collinear_data.csv: {mse:.4f}")
    print(f"R-squared for collinear_data.csv: {r2_score:.4f}")
    print(f"RMSE for collinear_data.csv: {rmse:.4f}")

    assert mse < 5.0, f"Mean squared error {mse} is too high"
    assert np.sum(np.abs(results.coef) < 1e-2) > 0, "Expected at least one coefficient to be near zero"

def test_new_test_data():
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "new_test_data.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    preds = results.predict(X)

    mse = np.mean((preds - y)**2)
    r2_score = 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(mse)

    print(f"\n new_test_data.csv")
    print(f"MSE for new_test_data.csv: {mse:.4f}")
    print(f"R-squared for new_test_data.csv: {r2_score:.4f}")
    print(f"RMSE for new_test_data.csv: {rmse:.4f}")

    assert mse < 1.0, f"Mean squared error {mse} is too high"

def test_large_test_data():
    data = []
    csv_path = os.path.join(os.path.dirname(__file__), "new_large_test_data.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])

    model = LassoHomotopyModel(tol=1e-4, max_iter=1000)
    results = model.fit(X, y)
    preds = results.predict(X)

    mse = np.mean((preds - y) ** 2)
    r2_score = 1 - (np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(mse)

    print(f"\n new_large_test_data.csv")
    print(f"MSE for new_large_test_data.csv: {mse:.4f}")
    print(f"R-squared for new_large_test_data.csv: {r2_score:.4f}")
    print(f"RMSE for new_large_test_data.csv: {rmse:.4f}")

    assert mse < 0.5, f"Mean squared error {mse} is too high"
