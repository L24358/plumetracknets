"""
References:
    - How to plot vectors: https://stackoverflow.com/questions/42281966/how-to-plot-vectors-in-python-using-matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 0
tpe = "constant"
nc = 3

# load
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=3.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"][:, :3] # (n_samples, n_components)
inn = pca.fit_transform(inn.T)

# plot
origin = np.mean(all_traj, axis=0)
origins = np.tile(origin, (nc,1))
V = pca.components_[:nc, :3] # (n_components, n_features)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(*all_traj.T, color="k", alpha=0.3)
ax.quiver(*origin.T, *inn.T, length=3, color=['r','g','b','r','r','g','g','b','b'])
ax.view_init(30, 70)
vis.gen_gif(True, "inn_components", ax, stall=5, angle1=30, angles=None)
