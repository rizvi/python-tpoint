import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)


# PLS using NIPALS algorithm
def nipals_algorithm(X, y, n_components, max_iters=100, tol=1e-4):
    n, p = X.shape
    T = np.zeros((n, n_components))
    P = np.zeros((p, n_components))
    W = np.zeros((p, n_components))

    for i in range(n_components):
        t = X[:, i].copy()
        t_old = np.zeros_like(t)

        for _ in range(max_iters):
            # Update weights
            w = X.T @ t
            w /= np.linalg.norm(w)

            # Update scores
            t = X @ w
            t /= np.linalg.norm(t)

            # Check for convergence
            if np.linalg.norm(t - t_old) < tol:
                break
            t_old = t.copy()

        # Update loadings
        p = X.T @ t / (t.T @ t)

        # Store results
        T[:, i] = t
        P[:, i] = p
        W[:, i] = w

        # Deflation
        X -= np.outer(t, p.T)

    # Regression coefficients
    B = np.linalg.pinv(T) @ y

    return T, P, W, B


# Number of components
n_components = 2

# Apply the NIPALS algorithm
T, P, W, B_pls = nipals_algorithm(X_train, y_train, n_components)

# Predict using the PLS model
y_pred_pls = X_test @ W @ B_pls

# Calculate MSE
mse_pls = mean_squared_error(y_test, y_pred_pls)

# Calculate R2 Score
r2_pls = r2_score(y_test, y_pred_pls)

print("PLS Regression Results:")
print(f"MSE: {mse_pls}")
print(f"R2 Score: {r2_pls}")
