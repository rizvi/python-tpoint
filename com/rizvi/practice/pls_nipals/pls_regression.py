import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

from com.rizvi.practice.pls_nipals.nipals_algo import NIPALS


# Define the PLS Regression class
class PLSRegression:
    def __init__(self, n_components, max_iters=100, tol=1e-4):
        # Initialize the NIPALS algorithm with the specified parameters
        self.nipals = NIPALS(n_components, max_iters, tol)
        self.n_components = n_components

    def fit(self, X, y):
        T, P, W = self.nipals.fit(X)  # Fit the NIPALS algorithm and get the score, loading, and weight matrices
        B = np.linalg.pinv(T) @ y  # Calculate the regression coefficients
        self.B = B  # Store the regression coefficients
        self.W = W  # Store the weight matrix
        return self

    def predict(self, X):
        return X @ self.W @ self.B  # Predict the target variable using the weight matrix and regression coefficients

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Extract the features
y = iris.target  # Extract the target variable

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardize the feature matrix

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)  # Encode the target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PLS Regression
n_components = 2  # Set the number of PLS components
pls = PLSRegression(n_components)  # Create a PLSRegression object
pls.fit(X_train, y_train)  # Fit the PLS regression model to the training data
y_pred_pls = pls.predict(X_test)  # Predict the target variable for the test data

# Calculate MSE
mse_pls = mean_squared_error(y_test, y_pred_pls)  # Calculate the mean squared error

# Calculate R2 Score
r2_pls = r2_score(y_test, y_pred_pls)  # Calculate the R-squared score

print("PLS Regression Results:")
print(f"MSE: {mse_pls}")
print(f"R2 Score: {r2_pls}")
