import numpy as np
def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:
	eigenvalues=np.linalg.eigvals(np.array(matrix))
	return[ float(eigenvalues[0]),float(eigenvalues[1])]