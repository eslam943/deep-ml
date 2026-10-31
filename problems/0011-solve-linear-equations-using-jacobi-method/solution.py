import numpy as np

def solve_jacobi(A: list, b: list, n: int) -> list:
    # Convert inputs to float NumPy arrays for full precision calculations
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Extract the diagonal elements
    D = np.diag(A)
    
    # Create a matrix containing only the off-diagonal elements (L + U)
    LU = A - np.diag(D)
    
    # Initialize the solution vector to all zeros
    x = np.zeros_like(b, dtype=float)
    
    # Perform exactly n iterations (sweeps)
    for _ in range(n):
        # Vectorized formula: x_new = (b - (L+U) * x_old) / D
        x = (b - np.dot(LU, x)) / D
        
    # Round only the final result to 4 decimal places and return as a standard list
    return np.round(x, 4).tolist()
