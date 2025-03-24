# Lasso Regression using the Homotopy Method

## Overview
This project implements **LASSO (Least Absolute Shrinkage and Selection Operator) regression** using the **Homotopy Method** from first principles. LASSO is a linear regression model that introduces L1 regularization, encouraging sparsity in the solution by shrinking some coefficients to zero. The Homotopy Method efficiently finds the entire LASSO path as the regularization parameter changes.

## Installation and Setup
To set up and run this project, follow these steps:

1. **Fork the Main repository from the URL**:
   ```sh
   Fork the Repo From This URL :- "https://github.com/Spring2025CS584/Project1.git"
   ```

2. **Clone the forked repository**:
   ```sh
   git clone "https://github.com/SPM587SP24/CS584_ML_Project1.git"
   cd Project1
   ```

3. **Create a virtual environment**:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

5. **Run tests**:
   ```sh
   cd .\LassoHomotopy\
   pytest tests/


   OR From Main Working Directory:
   pytest LassoHomotopy/tests/

   Run With Verbose:-
   pytest LassoHomotopy/tests/ -V

   ```

## Generating Test Data
To generate a working CSV file for testing, use the following command:
```sh
python generate_regression_data.py -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0 1 -seed 42 -output_file new_test_data.csv
```
This will create `new_test_data.csv`, which can be used for testing the LASSO model, Put that file in tests Directory with the other CSV files.

---

## Q&A
### 1. What does the model you have implemented do and when should it be used?
This model solves a **LASSO regression problem** using the **Homotopy Method**, which efficiently finds solutions as the regularization parameter \( \lambda \) changes.

**Use Cases:**
- When you need **feature selection** alongside regression.
- When you have **high-dimensional data** where some features may be irrelevant.
- When interpretability is important, since LASSO produces sparse solutions.
- When handling **collinear data**, as LASSO tends to select only one among highly correlated features.

### 2. How did you test your model to determine if it is working reasonably correctly?
The model was tested using:
- **Unit tests with synthetic data** (ensuring that the output is sparse where expected).
- **Comparison with SciKit Learn’s Lasso model** for validation.
- **Mean Squared Error (MSE) checks**, ensuring reasonable prediction accuracy.
- **Tests with highly collinear data** to verify that the model appropriately shrinks some coefficients to zero.

### 3. What parameters have you exposed to users of your implementation in order to tune performance?
The model allows users to set:
- **\( \lambda \) (lambda)**: The regularization strength, controlling the sparsity of the solution.
- **Tolerance**: Determines the stopping condition for convergence.
- **Max iterations**: Limits the number of steps to avoid long-running computations.

### 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
**Challenges:**
- **Extremely small \( \lambda \) values** can lead to overfitting and numerical instability.
- **Highly correlated features** may cause instability in coefficient selection.
- **Very high-dimensional data (p >> n)** might require additional optimizations.

**Potential Workarounds:**
- Implementing **cross-validation** to find an optimal \( \lambda \) value.
- Using **feature scaling and normalization** to improve numerical stability.
- Exploring **more efficient computational approaches** for large-scale datasets.

---

## Contributing
Contributions are welcome! Feel free to submit a pull request with improvements or fixes.

