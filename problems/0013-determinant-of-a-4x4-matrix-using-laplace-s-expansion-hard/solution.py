import numpy as np
def determinant_4x4(matrix: list[list[int|float]]) -> float:
	# Your recursive implementation here
	return np.linalg.det(matrix)