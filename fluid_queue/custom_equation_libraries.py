import numpy as np
import pysindy as ps

def min_library(N=10):
    def min_func(x):
        return np.minimum(x, N)

    def name_func(var):
        return f"min({var},{N})"

    return ps.CustomLibrary(
        library_functions=[min_func],
        function_names=[name_func]
    )
