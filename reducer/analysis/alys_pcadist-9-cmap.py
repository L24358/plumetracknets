"""
@ instructions:
    - to plot vfp with pos/neg graident colors:
        gifname = "pcadist_vfp"
        groups = [vangle_pos, vangle_neg]
        cmaps = ["Greens", "Reds"]
    - to plot r with gradient colors:
        gifname = "pcadist_r"
        groups = [rlength]
        cmaps = ["cool"]

@ references:
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
def concentration(info):
    vx, vy, C, r, theta = info # unpack
    return C > 0, C/Cmax

def vangle_pos(info):
    vx, vy, C, r, theta = info # unpack
    angle = np.arctan2(vy, vx)
    return angle > 0, abs(angle / np.pi)

def vangle_neg(info):
    vx, vy, C, r, theta = info # unpack
    angle = np.arctan2(vy, vx)
    return angle < 0, abs(angle / np.pi)

def vangle_0_2pi(info):
    vx, vy, C, r, theta = info # unpack
    angle = np.arctan2(vy, vx)
    pmask = angle >= 0
    nmask = angle < 0
    return True, (np.multiply(angle, pmask) + np.multiply(2*np.pi - angle, nmask)) / (2*np.pi) # normalize

def rlength(info):
    vx, vy, C, r, theta = info # unpack
    return r > 0, r / rmax

# hyperparameters
specify = 0
tpe = "constant"
gifname = "pcadist_r"
groups = [rlength]
cmaps = ["cool"]
assert len(groups) == len(cmaps)

# loading data
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca = pca_dic["pca"]
fp_pcas = bcs.npload("pcadist", f"fppca_agent={specify+1}_save=pca64.npy")
info = bcs.npload("pcadist", f"fppcainfo_agent={specify+1}_save=pca64.npy")

# calculate args
Cmax = max(info.T[2])
rmax = max(info.T[3])

# main: generate gifs
ax = plt.figure().add_subplot(projection="3d")
for g in range(len(groups)):
    # get labels
    fps, labels = [], []
    func = groups[g]
    for i in range(len(fp_pcas)):
        belong, label = func(info[i])
        if belong:
            fps.append(fp_pcas[i])
            labels.append(label)

    # plot
    coors = np.array(fps)[:,:3].T
    ax.scatter(*coors, c=labels, cmap=cmaps[g], s=3)
vis.gen_gif(True, gifname, ax, stall=5, angle1=30, angles=None)
