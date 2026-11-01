import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    reminder = n_samples % k
    base_size = n_samples // k
    
    indices = np.arange(n_samples)
    if shuffle == True:
        np.random.shuffle(indices)
        l = []
    else:
        l = []
    
    current_pos = 0
    for i in range(k):
        size = base_size + (1 if i < reminder else 0)
        l.append(indices[current_pos : current_pos + size].tolist())
        current_pos += size

    return [
        (
            [idx for fold_idx, fold in enumerate(l) if fold_idx != i for idx in fold],
            x
        )
        for i, x in enumerate(l)
    ]