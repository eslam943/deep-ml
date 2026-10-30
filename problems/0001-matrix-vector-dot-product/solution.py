import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) :
    a=np.array(a)
    b=np.array(b)
    if a.shape[1] != b.shape[0] :
        return -1
    else:
        return np.array( a@b)
        