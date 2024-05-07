# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :1]  # Using only the sepal length (first column)
y = iris.data[:, 2]   # Predicting the petal length (third column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the values for the testing data
predictions = model.predict(X_test)

# Visualize the model
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_test, predictions, color='red', label='Model prediction')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Linear Regression on Iris Dataset')
plt.legend()
plt.show()

# Print model performance
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
print('Coefficient of determination (R^2): %.2f' % r2_score(y_test, predictions))

# Print the coefficients
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
