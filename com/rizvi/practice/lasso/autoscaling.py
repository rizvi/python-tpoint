import numpy as np

# Sample data for illustration
height = np.array([65, 68, 70, 72, 74])  # Heights in inches
weight = np.array([140, 160, 180, 200, 220])  # Weights in pounds

# Calculate mean and standard deviation for each feature
height_mean = np.mean(height)
height_std = np.std(height)
weight_mean = np.mean(weight)
weight_std = np.std(weight)

# Autoscaling the features
autoscaled_height = (height - height_mean) / height_std
autoscaled_weight = (weight - weight_mean) / weight_std

# Display autoscaled values
print("Autoscaled height:", autoscaled_height)
print("Autoscaled weight:", autoscaled_weight)

# Output:
# Autoscaled height: [-1.53644256 -0.57616596  0.06401844  0.70420284  1.34438724]
# Autoscaled weight: [-1.41421356 -0.70710678  0.          0.70710678  1.41421356]

# In this example, we first calculate the mean and standard deviation for both height and weight.
# Then, we apply the autoscaling formula to each feature, subtracting the mean and dividing by the standard deviation.
#
# As a result, both the height and weight data are transformed such that they have a mean of 0 and a standard deviation of 1,
# which is a characteristic of autoscaled data.
# This preprocessing step can be beneficial for algorithms that are sensitive to the scale of the features,
# such as gradient descent-based optimization algorithms.