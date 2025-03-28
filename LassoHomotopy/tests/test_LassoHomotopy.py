import os
import csv
import numpy as np
from model.LassoHomotopy import LassoHomotopyModel

def test_predict():
    model = LassoHomotopyModel()
    data = []
    
    # Read data from CSV
    csv_path = os.path.join(os.path.dirname(__file__), "small_test.csv")
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})
    
    # Extract features (X) and target variable (y)
    X = np.array([[datum[k] for k in datum if k.startswith('x')] for datum in data])
    y = np.array([datum['y'] for datum in data])
    
    # Fit the model and make predictions
    results = model.fit(X, y)
    preds = results.predict(X)
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((preds - y) ** 2)
    
    # Compute R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - preds) ** 2)
    r2_score = 1 - (ss_residual / ss_total)
    
    # Compute RMSE
    rmse = np.sqrt(mse)
    
    # Print R-squared and RMSE
    print(f"R-squared: {r2_score:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Ensure MSE is within an acceptable threshold
    assert mse < 12, f"High MSE: {mse}, Predictions: {preds}, Actual: {y}"

