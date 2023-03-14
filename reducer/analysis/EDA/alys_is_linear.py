'''
Check if the matrix is close to operating in the linear regime.
'''

import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
from itertools import product

def sweep_observations(n, rnn, inn, br, bi):
    x1 = np.linspace(0, 1, n) # -2, 8
    x2 = np.linspace(-1, 1, n) # -4, 2
    x3 = np.linspace(-1, 1, n) # -0.2, 8

    grid = range(n)
    is_linear = np.zeros((n,n,n))
    for i, j, k in product(grid, grid, grid):
        x = np.array([x1[i], x2[j], x3[k]])
        Js = dy.get_jacobians(x, rnn, inn, br, bi)
        diffs = [bcs.different(J, rnn, threshold=np.mean(rnn)*0.01)[0] for J in Js]
        is_linear[i, j, k] = not np.any(diffs)

    return is_linear

if __name__ == "__main__":
    rnn, inn, br, bi = bcs.model_loader(specify=0) 
    is_linear = sweep_observations(5, rnn, inn, br, bi)
    print(is_linear)