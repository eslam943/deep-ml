# Cell 1
import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    # Calculate weights via Normal Equation
    theta = np.linalg.inv(X_arr.T @ X_arr) @ X_arr.T @ y_arr
    
    # Round to 4 decimal places and convert to a Python list
    return np.round(theta, 4).tolist()