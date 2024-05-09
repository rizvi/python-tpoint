import numpy as np

# Sample data for illustration
ages = np.array([30, 35, 40, 45, 50])  # Ages in years

# Calculate minimum and maximum values
min_age = np.min(ages)
max_age = np.max(ages)

# Min-max scaling
scaled_ages = (ages - min_age) / (max_age - min_age)

# Display scaled ages
print("Original ages:", ages)
print("Min-max scaled ages:")
print(scaled_ages)


# Output
# Original ages: [30 35 40 45 50]
# Min-max scaled ages:
# [0.   0.25 0.5  0.75 1.  ]