'''
Verify numerically solved fixed points by simulation.
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

# Fixed points for no input, no bias
x_0 = [0., 0., 0.]
args = [x_0, rnn, inn, np.zeros(64), np.zeros(64)]
fps = dy.get_fixed_points(*args)
stable = [dy.get_stability(fp, args) for fp in fps]

# simulate random initial conditions
rp = 9
ys = []
np.random.seed(100)
for _ in range(rp):
    h_0 = np.random.uniform(low=-1, high=1, size=(64,))
    _, y = dy.sim(rnn, inn, np.zeros(64), np.zeros(64), dy.constant_obs(x_0), h_0, T=100)
    ys.append(y)
ys_flatten = np.asarray(ys).reshape(-1, 64) # flatten across all i.c.

# perform PCA
pca = PCA(n_components=3)
y_pcas_flatten = pca.fit_transform(ys_flatten)
y_pcas = y_pcas_flatten.reshape(rp, -1, 3)
fp_pcas = pca.transform(fps)

# plot the trajectories
fig = plt.figure(figsize=(10,9))
for i in range(9): # only plot the first 9 trajectories
    ax = fig.add_subplot(3, 3, i+1, projection="3d")
    ax = vis.plot_trajectory(y_pcas[i].T, projection="3d", save=False, ax=ax)

    # plot the fixed points
    for fp_pca in fp_pcas:
        color = "k" if stable else "r"
        ax.scatter([fp_pca[0]], [fp_pca[1]], [fp_pca[2]], color=color)
vis.savefig(dpi=300)