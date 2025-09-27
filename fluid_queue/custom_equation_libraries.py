import numpy as np
import pysindy as ps

def min_library(N=2):
    def min_func(x):
        return np.minimum(x, N)

    funcs = [min_func]
    names = [
        lambda x: f"min(x,{N})"
    ]

    return ps.CustomLibrary(library_functions=funcs, function_names=names)
