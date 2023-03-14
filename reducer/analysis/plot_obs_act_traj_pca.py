"""
Observe how the location in the pca changes as a function of stimulus.
"""

import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 0
tpe = "constant"

# pca
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]

# append trajectories
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    hrnn = dic["activities_rnn"]
    observations = dic["observations"]
    actions = dic["actions"]
    y_pca = pca.transform(hrnn)

    # plot
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(241)
    vis.plot_quantities(observations.T, save=False, ax=ax1, ylabel="value", subtitle="Observations", color=["k", "r", "b"], label=["x", "y", "C"])

    ax2 = fig.add_subplot(242)
    traj, angles, actions = dy.get_trajectory2(actions)
    vis.plot_quantities(actions.T, save=False, ax=ax2, ylabel="value", subtitle="Actions", color=["g", "m"], label=["r", "\u03B8"])

    ax3 = fig.add_subplot(245)
    vis.plot_trajectory(traj.T, save=False, ax=ax3)
    
    ax4 = fig.add_subplot(122, projection="3d")
    ax4 = vis.plot_PCA_3d(None, y_pca=all_traj, plot_time=False, ax=ax4, save=False)
    ax4 = vis.plot_PCA_3d(None, y_pca=y_pca, ax=ax4, save=False)
    ax4.set_axis_off()

    vis.savefig(figname=bcs.fjoin("obs_act_traj_pca", f"agent={specify+1}_episode={episode}.png", tpe=2))
