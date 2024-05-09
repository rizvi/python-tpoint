import numpy as np

# Sample data for illustration
data = np.array([[1, 2, 3],
                 [2, 3, 4],
                 [3, 4, 5],
                 [4, 5, 6]])

# Calculate Euclidean norm for each data point
euclidean_norms = np.linalg.norm(data, axis=1)

# Scale each data point to have a Euclidean norm of one
scaled_data = data / euclidean_norms[:, np.newaxis]

# Display scaled data
print("Original data:")
print(data)
print("\nScaled data (after unilength scaling):")
print(scaled_data)

# OUTPUT:
# Original data:
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]]
#
# Scaled data (after unilength scaling):
# [[0.26726124 0.53452248 0.80178373]
#  [0.37139068 0.55708601 0.74278135]
#  [0.42426407 0.56568542 0.70710678]
#  [0.45584231 0.56980288 0.68376346]]