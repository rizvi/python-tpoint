import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Define the PLS Regression class
from com.rizvi.practice.pls_nipals_movielens.nipals_algo import NIPALS


# Define the PLS Regression class
class PLSRegression:
    def __init__(self, n_components, max_iters=100, tol=1e-4):
        self.nipals = NIPALS(n_components, max_iters, tol)
        self.n_components = n_components

    def fit(self, X, y):
        T, P, W = self.nipals.fit(X)
        B = np.linalg.pinv(T) @ y
        self.B = B
        self.W = W
        return self

    def predict(self, X):
        return X @ self.W @ self.B

# Load the MovieLens dataset
# Assuming you have downloaded the MovieLens 100K dataset and placed it in the 'ml-100k' directory
df = pd.read_csv('ml-100k/u.data', delimiter='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])

# Prepare the data
X = df[['user_id', 'item_id']].values
y = df['rating'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode the target variable (ratings)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and test sets
# With such a small test size, ensure that the split is stratified
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, random_state=42, stratify=y)

# Check if the test set is empty and adjust accordingly
if len(y_test) == 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / len(y), random_state=42, stratify=y)

# Apply PLS Regression
n_components = 2
pls = PLSRegression(n_components)
pls.fit(X_train, y_train)
y_pred_pls = pls.predict(X_test)

# Calculate MSE
mse_pls = mean_squared_error(y_test, y_pred_pls)

# Calculate R2 Score
r2_pls = r2_score(y_test, y_pred_pls)

print("PLS Regression Results:")
print(f"MSE: {mse_pls}")
print(f"R2 Score: {r2_pls}")
