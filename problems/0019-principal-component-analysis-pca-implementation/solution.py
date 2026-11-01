import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    # 1. Standardize dataset column-wise
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_std = (data - mean) / std

    # 2. Compute covariance matrix across features
    cov_matrix = np.cov(data_std, rowvar=False)

    # Handle 1D covariance matrix if data has only 1 feature
    cov_matrix = np.atleast_2d(cov_matrix)

    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # 5. Select top k principal components
    top_k_components = sorted_eigenvectors[:, :k]

    # 6. Apply sign convention per eigenvector column
    for col in range(top_k_components.shape[1]):
        vec = top_k_components[:, col]
        # Find first element with absolute value > 1e-10
        nonzero_idx = np.where(np.abs(vec) > 1e-10)[0]
        if len(nonzero_idx) > 0:
            first_idx = nonzero_idx[0]
            if vec[first_idx] < 0:
                top_k_components[:, col] *= -1

    # 7. Round output to 4 decimal places
    return np.round(top_k_components, 4)