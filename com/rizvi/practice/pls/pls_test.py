from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate some synthetic data for demonstration
X, y = make_regression(n_samples=10, n_features=5, noise=0.5)
print("X data is, ", X)
print("y data is, ", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize PLS regression
pls = PLSRegression(n_components=2)

# Fit the model on the training data
pls.fit(X_train, y_train)

# Predict on the test data
y_pred = pls.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)