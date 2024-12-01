import numpy as np
from regularize_lin_reg import LinearRegression
import matplotlib.pyplot as plt

# Load the data from the file
data = np.genfromtxt('men-olympics-100.txt')
years = data[:, 0]
times = data[:, 1]

# Prepare log-spaced lambda values
lambdas = np.logspace(-8, 0, 100, base=10)

# Initialize variables to store results
errors = []
best_lambda = None
lowest_error = float('inf')
best_coeffs_no_reg = None
best_coeffs_reg = None

# Polynomial degree for part (a)
degree = 3

# Perform leave-one-out cross-validation
for lambda_ in lambdas:
    loo_errors = []
    for i in range(len(years)):
        X_train = np.delete(years, i).reshape(-1, 1)
        t_train = np.delete(times, i).reshape(-1, 1)
        X_val = years[i].reshape(-1, 1)
        t_val = times[i]
        
        # Train the model
        model = LinearRegression(lambda_=lambda_, degree=degree)
        model.fit(X_train, t_train)
        
        # Validate the model
        prediction = model.predict(X_val)
        error = (prediction - t_val) ** 2
        loo_errors.append(error.item())
    
    # Calculate mean error for this lambda
    mean_error = np.mean(loo_errors)
    errors.append(mean_error)
    
    # Update best lambda if necessary
    if mean_error < lowest_error:
        lowest_error = mean_error
        best_lambda = lambda_
        best_coeffs_reg = model.w

# Fit the model without regularization (lambda = 0)
model_no_reg = LinearRegression(lambda_=0, degree=degree)
model_no_reg.fit(years.reshape(-1, 1), times.reshape(-1, 1))
best_coeffs_no_reg = model_no_reg.w

# Plot the errors vs lambda
# plt.figure(figsize=(10, 6))
# plt.semilogx(lambdas, errors, marker='o', linestyle='-', color='b')
# plt.xlabel('Lambda (log scale)')
# plt.ylabel('Leave-One-Out Cross-Validation Error')
# plt.title('Leave-One-Out Cross-Validation Error vs Lambda')
# plt.grid(True)
# plt.show()

# Store results for part (a)
part_a_results = {
    "best_lambda": best_lambda,
    "lowest_error": lowest_error,
    "coefficients_no_reg": best_coeffs_no_reg.flatten(),
    "coefficients_reg": best_coeffs_reg.flatten(),
}

# Output results for part (a)
print(part_a_results)

# Polynomial degree for part (b)
degree = 4

# Initialize variables for part (b)
errors_b = []
best_lambda_b = None
lowest_error_b = float('inf')
best_coeffs_no_reg_b = None
best_coeffs_reg_b = None

# Perform leave-one-out cross-validation for degree 4
for lambda_ in lambdas:
    loo_errors = []
    for i in range(len(years)):
        X_train = np.delete(years, i).reshape(-1, 1)
        t_train = np.delete(times, i).reshape(-1, 1)
        X_val = years[i].reshape(-1, 1)
        t_val = times[i]
        
        # Train the model
        model = LinearRegression(lambda_=lambda_, degree=degree)
        model.fit(X_train, t_train)
        
        # Validate the model
        prediction = model.predict(X_val)
        error = (prediction - t_val) ** 2
        loo_errors.append(error.item())
    
    # Calculate mean error for this lambda
    mean_error = np.mean(loo_errors)
    errors_b.append(mean_error)
    
    # Update best lambda if necessary
    if mean_error < lowest_error_b:
        lowest_error_b = mean_error
        best_lambda_b = lambda_
        best_coeffs_reg_b = model.w

# Fit the model without regularization (lambda = 0)
model_no_reg_b = LinearRegression(lambda_=0, degree=degree)
model_no_reg_b.fit(years.reshape(-1, 1), times.reshape(-1, 1))
best_coeffs_no_reg_b = model_no_reg_b.w

# Plot the errors vs lambda for degree 4
plt.figure(figsize=(10, 6))
plt.semilogx(lambdas, errors_b, marker='o', linestyle='-', color='r')
plt.xlabel('Lambda (log scale)')
plt.ylabel('Leave-One-Out Cross-Validation Error')
plt.title('Leave-One-Out Cross-Validation Error vs Lambda (Degree 4)')
plt.grid(True)
plt.show()

# Store results for part (b)
part_b_results = {
    "best_lambda": best_lambda_b,
    "lowest_error": lowest_error_b,
    "coefficients_no_reg": best_coeffs_no_reg_b.flatten(),
    "coefficients_reg": best_coeffs_reg_b.flatten(),
}

# Output results for part (b)
print(part_b_results)
