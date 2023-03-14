'''
Establish that fixed points are the only structures driving the system. (Quantitatively)
'''

import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# Load model
specify = 0
rnn, inn, br, bi = bcs.model_loader(specify=specify)
obs_unique = bcs.npload("observations", "random_select.npy")

# set seed
np.random.seed(42)

# set recurrent function
def get_convergence(x_0, fps, h_0=[], T=200, iteration=1):
    if len(h_0) == 0: h_0 = np.random.uniform(low=-1, high=1, size=(64,)) # if h_0 is not passed in
    _, y = dy.sim(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, T=T)

    for fp in fps:
        is_diff, _ = bcs.different(fp, y[-1], threshold=1e-2)
        if not is_diff: return True, h_0, y[-1] # converges

    if iteration > 3: return False, h_0, y[-1] # if exceeds max_iter=3, return not converge
    return get_convergence(x_0, fps, h_0=h_0, T=T+1000, iteration=iteration+1)

if __name__ == "__main__":
    # simulate random initial conditions
    rp = 10
    for i in range(429, len(obs_unique)):

        print(i)
        x_0 = obs_unique[i]
        args = [x_0, rnn, inn, br, bi]
        fps = dy.get_fixed_points(*args)

        for _ in range(rp):
            converge, h_0, y_last = get_convergence(x_0, fps)
            if not converge:
                print("Did not converge!")
                to_save = {"x_0": x_0, "fps": fps, "h_0": h_0, "y_last": y_last}
                bcs.pklsave(to_save, "fixed_points_random", f"agent={specify+1}_idx={i}.pkl")
