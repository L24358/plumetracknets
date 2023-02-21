"""
Test if in tracking, the only stable structure is the fixed points.
"""

import os 
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
episode = 38

# model loading
start = bcs.track_dic_manual[episode]
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=3.pkl")
pca = pca_dic["pca"]
dic = bcs.simulation_loader(specify, tpe, episode=episode, verbose=False)
hs = dic["activities_rnn"]

# sample from time to get fixed points
for t in np.linspace(start, len(hs)-1, 10):
    x_0 = dic["observations"][int(t)]
    h_0 = dic["activities_rnn"][int(t)]

    # fixed points and stability
    args = [x_0, rnn, inn, br, bi]
    fps = dy.get_fixed_points(*args, rp=100)
    stable = [dy.get_stability(fp, args) for fp in fps]
    fp_pcas = pca.transform(fps)

    # plot with logged h_0
    ax = plt.figure().add_subplot(111, projection="3d")
    _, hrnn = dy.sim(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, T=100)
    ax = vis.plot_trajectory(pca.transform(hrnn).T, save=False, ax=ax)

    # plot with random h_0
    for _ in range(10):
        h_0 = np.random.uniform(low=-1, high=1, size=(64,))
        _, hrnn = dy.sim(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, T=100)
        ax = vis.plot_trajectory(pca.transform(hrnn).T, save=False, ax=ax)

    # plot fixed points
    for i in range(len(fp_pcas)):
        color = "k" if stable[i] else "r"
        ax.scatter([fp_pcas[i][0]], [fp_pcas[i][1]], [fp_pcas[i][2]], color=color)
    fname = bcs.fjoin("tracking_fps", f"agent={specify+1}_epi={episode}_t={int(t)}.png", tpe=2)
    vis.savefig(figname=fname)


