"""
Plot fixed point agent angle a_theta(h^*) as a function of wind direction phi. Sorted by regimes.

References:
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
c1, c2 = 5, 12

# loading data
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca = pca_dic["pca"]
fp_pcas = bcs.npload("pcadist", f"fppca_agent={specify+1}_save=pca64.npy")
info = bcs.npload("pcadist", f"fppcainfo_agent={specify+1}_save=pca64.npy")

# get angles and fixed point actions
vx, vy, C, _, _, epis, ts = info.T
angles = np.arctan2(vy, vx)
fps = pca.inverse_transform(fp_pcas)
fp_actions = dy.get_action_from_h(specify, fps, return_info=False, transform=True)
fp_rs, fp_thetas = fp_actions.T
dic_regime = bcs.pklload("regimes", f"regime_agent={specify+1}_criterion=({c1},{c2}).pkl")

# calculate and define
ival = np.pi / (gsize//2)
plus = gsize//2 - 1
wind_grid = np.linspace(-np.pi, np.pi, gsize)
wind_grid_mean = (wind_grid[1:] + wind_grid[:-1]) / 2
means = {"tracking": {}, "recovery": {}, "lost": {}}
idxs = {"tracking": [], "recovery": [], "lost": []}
for key in means.keys():
    for i in range(gsize - 1): means[key][i] = []
def regime(epi, t):
    epi, t = int(epi), int(t)
    for reg in ["tracking", "recovery", "lost"]:
        if dic_regime[epi][reg][t]: return reg
    raise bcs.AlgorithmError()

# sort and average
target = fp_thetas
for i in range(len(C)):
    angle, theta, epi, t = angles[i], target[i], epis[i], ts[i]
    category = np.sign(angle)*(abs(angle) // ival) + plus
    reg = regime(epi, t)
    
    means[reg][int(category)].append(theta)
    idxs[reg].append(i)
mean = [np.mean(means["tracking"][i]) for i in range(gsize - 1)]
mean2 = [np.mean(means["recovery"][i]) for i in range(gsize - 1)]

# plot
ax = plt.figure().add_subplot(111)
vis.simpleaxis(ax)
plt.scatter(angles[idxs["tracking"]], target[idxs["tracking"]], s=1, alpha=0.5, color="lightgreen")
plt.scatter(angles[idxs["recovery"]], target[idxs["recovery"]], s=1, alpha=0.3, color="lightsteelblue")
plt.scatter(angles[idxs["lost"]], target[idxs["lost"]], s=1, alpha=0.3, color="pink")
plt.plot(wind_grid_mean, mean, color="g", marker='o')
plt.plot(wind_grid_mean, mean2, color="b", marker='o')
plt.plot([-np.pi/2, -np.pi/2], [min(target[idxs["tracking"]]), max(target[idxs["tracking"]])], "k--") 
plt.plot([np.pi/2, np.pi/2], [min(target[idxs["tracking"]]), max(target[idxs["tracking"]])], "k--")
plt.xlabel("wind angle"); plt.ylabel("agent angle $a(h^*)$")
vis.savefig(figname=f"fpangle_windangle_regime_criterion=({c1},{c2}).png")
