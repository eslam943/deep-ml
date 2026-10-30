# Cell 1
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    import numpy as np
    matrix=np.array(matrix)
    means=[]
    if mode== 'column':
        for i in np.arange(matrix.shape[1]):
            means.append(np.mean(matrix[:,i]))
    else:
        for i in np.arange(matrix.shape[0]):
            means.append(np.mean(matrix[i,:]))
    return means