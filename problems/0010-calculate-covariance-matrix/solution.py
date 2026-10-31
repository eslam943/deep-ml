import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	vectors=np.array(vectors)
	return np.cov(vectors)