# Cell 1
import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    
    A=np.array(A)
    T=np.linalg.inv(np.array(T))
    S=np.array(S)
    if np.linalg.det(T)!=0 and np.linalg.det(S)!=0:
        transformed_matrix=T@A@S
        return transformed_matrix
    else:
        return -1