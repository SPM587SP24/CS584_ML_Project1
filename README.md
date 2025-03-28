# Lasso Regression using the Homotopy Method


## Overview
This project implements **LASSO (Least Absolute Shrinkage and Selection Operator) regression** using the **Homotopy Method** from first principles. LASSO is a linear regression model that introduces L1 regularization, encouraging sparsity in the solution by shrinking some coefficients to zero. The Homotopy Method efficiently finds the entire LASSO path as the regularization parameter changes.


### Code Explanation
The implementation is centered on the `LassoHomotopyModel` class which follows a simplified homotopy (LARS-like) method:
- **Initialization:**  
  The model accepts optional tuning parameters such as tolerance (`tol`) and maximum iterations (`max_iter`) and initializes the coefficients to zero.
- **Fitting Procedure:**  
  - The algorithm starts by computing the maximum absolute correlation between the predictors and the target, which is used as the initial regularization parameter (\( \lambda \)).
  - In each iteration, it identifies the predictor most correlated with the current residual and adds it to the active set.
  - The model then solves the least squares problem restricted to the active set to update the coefficients.
  - The residual is recalculated and \( \lambda \) is updated based on the new correlations.
  - This process repeats until \( \lambda \) falls below the specified tolerance or the maximum number of iterations is reached.
- **Prediction:**  
  After fitting, the `LassoHomotopyResults` class provides a `predict` method to compute outputs for new inputs based on the learned coefficients.


### Test Cases Overview
Three main test files are provided to validate the model:
1. **small_test.csv:**  
   - Checks basic model performance on a small dataset.
   - Evaluates metrics such as R-squared and RMSE to ensure that the Mean Squared Error (MSE) is below a specified threshold.
2. **collinear_data.csv:**  
   - Tests the model on data with highly collinear features.
   - Verifies that the model yields a sparse solution by ensuring that at least one coefficient is near zero.
   - Also checks that prediction accuracy remains within acceptable bounds.
3. **new_test_data.csv:**  
   - Ensures that the model performs well on a different test set with a smaller number of features.
   - Again, the MSE is checked to ensure that predictions are accurate.
4. **new_large_test_data.csv:**  
   - Tests the model on a larger dataset with a greater number of features.
   - Evaluates how well the algorithm scales and whether it continues to produce accurate predictions with increasing complexity.
   - Ensures computational efficiency while maintaining accuracy.

---

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

   Run with verbose output:
   pytest LassoHomotopy/tests/ -s -V

   ```

## Generating Test Data
To generate a working CSV file for testing, use the following command:
```sh
python generate_regression_data.py -N 100 -m 1.0 2.0 -b 0.5 -scale 0.1 -rnge 0 1 -seed 42 -output_file new_test_data.csv
```
This will create `new_test_data.csv`, which can be used for testing the LASSO model, Put that file in tests Directory with the other CSV files.

To generate a Large working CSV file for testing, use the following command:
```sh
python generate_regression_data.py -N 2000 -m 1.0 1.5 2.0 2.5 
3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 -b 0.5 -scale 0.1 -rnge 0 1 -seed 42 -output_file new_large_test_data.csv
```
This will create `new_large_test_data.csv`, which can be used for testing the LASSO model, Put that file in tests Directory with the other CSV files.

---

## Evaluation Metrics
The model's performance is evaluated using the following metrics:

- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted values. A lower MSE indicates better model performance.
  \[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

- **R-Squared (R²):** Indicates how well the model explains the variability of the target variable. A value closer to 1 means better model performance.
  \[ R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} \]

- **Root Mean Squared Error (RMSE):** The square root of MSE, providing an interpretable metric in the same unit as the target variable.
  \[ RMSE = \sqrt{MSE} \]

---

## 📓 Notebook Walkthrough: Detailed LASSO Homotopy Experiments

This notebook walks through a complete implementation of **LASSO Regression using the Homotopy Method**, with applications to multiple datasets. Each dataset tests different properties — from feature selection to performance with collinearity and real-world data. Additional experiments include synthetic data and SHAP explainability.

---

### 🔍 Dataset 1: `small_test.csv`

**Structure:**  
- **Samples:** 50  
- **Features:** 3 (`x_0`, `x_1`, `x_2`)  
- **Target:** `y`

**Purpose:** Test basic functionality of the Homotopy model on a small, clean dataset.

**Behavior:**  
- All features retained.
- No sparsity enforced due to all features contributing.
- Coefficients remain non-zero throughout the path.

**Evaluation:**  
- MSE ≈ 11.6  
- R² ≈ 0.98  

---

### 🔍 Dataset 2: `collinear_data.csv`

**Structure:**  
- **Samples:** 1000  
- **Features:** 10 (`X_1` to `X_10`)  
- **Target:** `target`

**Purpose:** Validate how the model handles multicollinearity.

**Behavior:**  
- LASSO drops redundant features.
- Only a sparse subset retained.
- Reflects LASSO’s strength in feature selection.

**Evaluation:**  
- MSE ≈ 4.06  
- R² ≈ 0.84  
- Sparsity: Yes

---

### 🔍 Dataset 3: `new_test_data.csv`

**Structure:**  
- **Samples:** 100  
- **Features:** 2 (`x_0`, `x_1`)  
- **Target:** `y`

**Purpose:** Demonstrate simple, low-noise linear regression.

**Behavior:**  
- Both features retained.
- Coefficients accurately reflect feature importance.

**Evaluation:**  
- MSE ≈ 0.0419  
- R² ≈ 0.91  
- Sparsity: No (both features are relevant)

---

### 🔍 Dataset 4: `new_large_test_data.csv`

**Structure:**  
- **Samples:** 2000  
- **Features:** 15 (`x_0` to `x_14`)  
- **Target:** `y`

**Purpose:** Stress test for scalability and feature sparsity.

**Behavior:**  
- Only meaningful features retained.
- Coefficients of irrelevant features shrunk to zero.

**Evaluation:**  
- MSE ≈ 0.0157  
- R² ≈ 0.9995  
- Sparsity: High

---

### 🧪 Synthetic Dataset

**Setup:**  
- 100 samples, 10 features.
- True coefficient vector known.
- `y` created with noise added.

**Behavior:**  
- Model accurately recovers correct non-zero coefficients.
- Demonstrates correctness of implementation.

**Visuals:**  
- Stem plot of learned coefficients vs true
- Prediction vs actual plot

---

### 🍷 Wine Quality Dataset (`WineQT.csv`)

**Setup:**  
- Real-world dataset with 11 features.
- Target: Wine quality score (integer)

**Steps:**  
- Standardization + normalization
- Cross-validation to find best λ
- Predictions rounded and saved
- SHAP explainability plots created

**Evaluation:**  
- Metrics: MSE, RMSE, R² printed
- SHAP summary and dependence plots saved

---

### 💉 Diabetes Dataset (`sklearn.datasets.load_diabetes`)

**Setup:**  
- Built-in real-world dataset from sklearn

**Steps:**  
- Model trained and evaluated on full data
- Coefficient sparsity and predictions analyzed

**Evaluation:**  
- MSE printed
- Coefficient plot shows LASSO shrinkage effect

---

## 📌 Summary

| Dataset                  | Samples | Features | Purpose                         | MSE       | RMSE     | R²       |
|--------------------------|---------|----------|----------------------------------|-----------|----------|----------|
| `small_test.csv`         | 50      | 3        | Basic regression validation      | ~11.61    | ~3.41    | ~0.98    |
| `collinear_data.csv`     | 1000    | 10       | Handles multicollinearity        | ~4.06     | ~2.02    | ~0.84    |
| `new_test_data.csv`      | 100     | 2        | Simple 2-feature test            | ~0.042    | ~0.20    | ~0.91    |
| `new_large_test_data.csv`| 2000    | 15       | High-dimensional regression      | ~0.0157   | ~0.1252  | ~0.9995  |
| `WineQT.csv`             | ~1000   | 11       | Real-world regression + SHAP     | ~0.58     | ~0.76    | ~0.46    |
| `Diabetes`               | 442     | 10       | Real-world regression            | ~2859.70  | ~53.48   | ~0.52    |

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


## Contributors: -
- **Neel Patel (A20524638) - npatel157@hawk.iit.edu**
- **Karan Savaliya (A20539487) - ksavaliya@hawk.iit.edu**
- **Deep Patel (A20545631) - dpatel224@hawk.iit.edu**
- **Johan Vijayan (A20553527) - jvijayan1@hawk.iit.edu**

Additional Contributions are welcome! Feel free to submit a pull request with improvements or fixes.

---


### References

- **Primary Paper: LASSO Homotopy Method (NIPS 2008)**
  - **Description:** This paper, along with its references, guided the implementation of the Homotopy method for LASSO regression.
  - **Link:** [https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf)

- **Regression Shrinkage and Selection via the Lasso - Tibshirani (1996)**
  - **Contains:** LASSO formulation, optimization strategies, and an early approach to solving LASSO.
  - **Link:** [https://doi.org/10.2307/2346178](https://doi.org/10.2307/2346178)  
    *(Alternatively, see [https://projecteuclid.org/euclid.ss/1038427203](https://projecteuclid.org/euclid.ss/1038427203))*
  - **Section:** Algorithmic description is in Section 3, discussing the coordinate descent and penalty function.

- **Homotopy Algorithm for LASSO - Osborne et al. (2000)**
  - **Contains:** Detailed mathematical formulation of the homotopy algorithm used for LASSO.
  - **Link:** [https://www2.isye.gatech.edu/~presnell/draps.pdf](https://www2.isye.gatech.edu/~presnell/draps.pdf)
  - **Section:** Algorithm in Section 4, describing how lambda is updated iteratively, similar to the implementation in this project.

- **Additional Reference:**
  - This project’s code is also informed by various open-source implementations and academic resources on LASSO regression and the Homotopy Method.


