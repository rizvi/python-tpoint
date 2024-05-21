import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("iris data: ", X)
print("iris target: ", y)

# Center and scale the predictor variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("X_scaled: ", X_scaled)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Perform PLSR with increasing number of components
n_components = min(X_train.shape[1], X_train.shape[0])
print("n_components: ", n_components)
mse_list = []
r2_list = []

for n in range(1, n_components + 1):
    # Create a PLSR model
    pls = PLSRegression(n_components=n)

    # Fit the model to the training data
    pls.fit(X_train, y_train)

    # Predict the response variable for the test data
    y_pred = pls.predict(X_test)

    # Calculate mean squared error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append the results to the lists
    mse_list.append(mse)
    r2_list.append(r2)

# Find the optimal number of components based on the minimum MSE
optimal_n = np.argmin(mse_list) + 1
optimal_mse = np.min(mse_list)
optimal_r2 = r2_list[optimal_n - 1]

print(f"Optimal number of components: {optimal_n}")
print(f"Minimum MSE: {optimal_mse}")
print(f"R-squared: {optimal_r2}")
