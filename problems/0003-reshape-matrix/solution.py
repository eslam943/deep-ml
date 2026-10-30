# Cell 1
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    #Write your code here and return a python list after reshaping by using numpy's tolist() method
    a=np.array(a)
    if a.shape[0]+a.shape[1] != 0 and  a.shape[0]*a.shape[1]==new_shape[0]*new_shape[1] :
        return np.reshape(a,new_shape).tolist()
    else:
        return []