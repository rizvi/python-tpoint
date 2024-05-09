import numpy as np

# Example vector
x = np.array([3, 4])

# Calculate Euclidean norm
euclidean_norm = np.linalg.norm(x)

# Display Euclidean norm
print("Euclidean norm of vector", x, ":", euclidean_norm)


#OUTPUT:
# Euclidean norm of vector [3 4] : 5.0
# In this example:
#
#     We have a 2-dimensional vector x=[3,4].
#     We use np.linalg.norm() function to calculate the Euclidean norm of x.
    # The Euclidean norm of x is calculated as 3*3 + 4*4 = 9+16 = 25. root of 25 = 5



