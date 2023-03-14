'''
Establish that fixed points are the only structures driving the system. (Quantitatively)
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA
from verify_fixed_point_dynamics2 import get_convergence

# Load model
specify = 0
rnn, inn, br, bi = bcs.model_loader(specify=specify)
name1, name2 = "random_select", "random" # "unique", "unique"
obs_unique = bcs.npload("observations", name1 + ".npy")

# functions
def num_converged(a, b, threshold=1e-1):
    d = abs(a-b)
    return (d < threshold).sum()

def num_converged_wrap(fps, y_last):
    n = [num_converged(fp, y_last) for fp in fps]
    return max(n)

# percentage
obs = []
for f in sorted(os.listdir(f"/src/data/fixed_points_{name2}")):
    data = bcs.pklload(f"fixed_points_{name2}", f)
    obs.append(tuple(data["x_0"]))
N = len(list(set(obs)))
print("Total number: ", N)
print("Percentage: ", N/5000.0)

# plot
pca = PCA(n_components=3)
for f in os.listdir(f"/src/data/fixed_points_{name2}"):
    data = bcs.pklload(f"fixed_points_{name2}", f)
    
    hrnn = dy.sim(rnn, inn, br, bi, dy.constant_obs(data["x_0"]), data["h_0"], T=200)[1]
    y_pca = pca.fit_transform(hrnn)
    ax = plt.figure().add_subplot(111)
    vis.plot_trajectory(y_pca.T, projection="3d", save=False, ax=ax)
    vis.savefig(figname=name2 + "_" + f[:-3] + "png")

