# Cell 1
import numpy as np
from sklearn.linear_model import LinearRegression
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # 1. Initialize LinearRegression with fit_intercept=False 
    # (because X already includes the column of 1s for bias)
    model = LinearRegression(fit_intercept=False)
    
    # 2. Fit the model
    model.fit(X, y)
    
    # 3. Extract coefficients, round to 4 decimals, and convert to list
    coefficients = np.round(model.coef_, 4).tolist()
    
    return coefficients