import numpy as np

# Sample data for illustration
age = np.array([30, 35, 40, 45, 50])  # Ages in years
income = np.array([50000, 60000, 70000, 80000, 90000])  # Incomes in dollars

# Calculate mean for each feature
age_mean = np.mean(age)
income_mean = np.mean(income)

# Centering the features
centered_age = age - age_mean
centered_income = income - income_mean

# Display centered values
print("Centered age:", centered_age)
print("Centered income:", centered_income)

# OUTPUT:
# Centered age: [-10.  -5.   0.   5.  10.]
# Centered income: [-20000. -10000.      0.  10000.  20000.]

# In this example, we first calculate the mean of age and income. Then, we subtract these means from every data point in the respective feature.
#
# After centering:
#
#     The mean of the centered age is 0.
#     The mean of the centered income is also 0.
#
# Centering doesn't affect the spread or distribution of the data; it only shifts the data distribution to be centered around 0.
# This preprocessing step can be useful in certain algorithms or analyses where it's desirable to remove the influence of the mean from the data.