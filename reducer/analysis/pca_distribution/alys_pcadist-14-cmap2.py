"""
Similar to pcadist-9-cmap, except this one plots fixed point actions.

Instructions:
    - to plot r with gradient colors:
        gifname = "pcadist_r"
        groups = [rlength]
        cmaps = ["cool"]
    - to plot theta with pos/neg graident colors:
        gifname = "pcadist_theta"
        groups = [tangle_pos, tangle_neg]
        cmaps = ["Blues", "RdPu"]

References:
    - Matplotlib colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# define the label groups and colors
def tangle_pos(action, thre=1e-04):
    r, theta = action
    return theta > thre, abs(theta / np.pi)

def tangle_neg(action, thre=1e-04):
    r, theta = action
    return theta < -thre, abs(theta / np.pi)

def tangle_zero(action, thre=1e-04):
    r, theta = action
    return abs(theta) <= thre, 0.5

def rlength(action):
    r, theta = action
    return r > 0, r / rmax

# hyperparameters
specify = 0
tpe = "constant"
gifname = "pcadist_theta"
groups = [tangle_pos, tangle_neg]
cmaps = ["Blues", "RdPu"]
assert len(groups) == len(cmaps)

# loading data
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca = pca_dic["pca"]
fp_pcas = bcs.npload("pcadist", f"fppca_agent={specify+1}_save=pca64.npy")
info = bcs.npload("pcadist", f"fppcainfo_agent={specify+1}_save=pca64.npy")
hs = pca.inverse_transform(fp_pcas)
actions = dy.get_action_from_h(specify, hs, return_info=False, transform=True)

# calculate args
Cmax = max(info.T[2])
rmax = max(info.T[3])

# main: generate gifs
ax = plt.figure().add_subplot(projection="3d")
for g in range(len(groups)):
    # get labels
    fps, labels = [], []
    func = groups[g]
    for i in range(len(actions)):
        belong, label = func(actions[i])
        if belong:
            fps.append(fp_pcas[i])
            labels.append(label)

    # plot
    coors = np.array(fps)[:,:3].T
    if cmaps[g] != None: ax.scatter(*coors, c=labels, cmap=cmaps[g], s=3)
    else: ax.scatter(*coors, color=labels, s=3)
vis.gen_gif(True, gifname, ax, stall=5, angle1=30, angles=None)
