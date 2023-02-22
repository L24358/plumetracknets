"""
@ references:
    - Matplotlib colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
"""

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 0
tpe = "constant"
gsize = 16
plot_ex = True

# loading data
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca = pca_dic["pca"]
fp_pcas = bcs.npload("pcadist", f"fppca_agent={specify+1}_save=pca64.npy")
info = bcs.npload("pcadist", f"fppcainfo_agent={specify+1}_save=pca64.npy")

# get angles and fixed point actions
vx, vy, C = info.T[:3]
idxs = np.where(C > 0)[0]
idxs_ex = np.setdiff1d(np.array(range(len(C))), idxs) # those that are not included in idxs
angles = np.arctan2(vy, vx)
fps = pca.inverse_transform(fp_pcas)
fp_actions = dy.get_action_from_h(specify, fps, return_info=False, transform=True)
fp_rs, fp_thetas = fp_actions.T

# sort and average
ival = np.pi / (gsize//2)
plus = gsize//2 - 1
means = {}
for i in range(gsize - 1): means[i] = []
for i in idxs:
    angle, r = angles[i], fp_rs[i]
    category = np.sign(angle)*(abs(angle) // ival) + plus
    means[int(category)].append(r)
mean = [np.mean(means[i]) for i in range(gsize - 1)]
wind_grid = np.linspace(-np.pi, np.pi, gsize)
wind_grid_mean = (wind_grid[1:] + wind_grid[:-1]) / 2

ax = plt.figure().add_subplot(111)
vis.simpleaxis(ax)
plt.scatter(angles[idxs], fp_rs[idxs], s=1, alpha=0.3, color="lightsteelblue")
if plot_ex: plt.scatter(angles[idxs_ex], fp_rs[idxs_ex], s=1, alpha=0.3, color="grey")
plt.plot(wind_grid_mean, mean, color="b", marker='o')
plt.xlabel("wind angle"); plt.ylabel("agent speed $a(h^*)$")
vis.savefig(figname=f"fpspeed_windangle_plotex={int(plot_ex)}.png")
