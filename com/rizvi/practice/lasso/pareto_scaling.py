import numpy as np

# Sample data for illustration
incomes = np.array([30000, 40000, 50000, 60000, 70000])  # Incomes in dollars

# Calculate mean and standard deviation
mean_income = np.mean(incomes)
std_income = np.std(incomes)

# Pareto scaling
scaled_incomes = (incomes - mean_income) / np.sqrt(std_income)

# Display scaled incomes
print("Original incomes:", incomes)
print("Pareto scaled incomes:")
print(scaled_incomes)


# Output
# Original incomes: [30000 40000 50000 60000 70000]
# Pareto scaled incomes:
# [-168.17928305  -84.08964153    0.           84.08964153  168.17928305]