'''
Plot fixed points in terms of the instantaneous inputs, evolving in time. Also plots sign(wind angle) and sign(agent angle).
'''

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA
from reducer.support.basics import constant, single_sine
from reducer.config import modelpath, graphpath

# parameters
start_dic = {38: 30, 16: 0, 21:20, 91:50}
simulate_fp = False
specify = 0
episode = 16
start = start_dic[episode]
foldername = f"fixed_point_t_episode={episode}"

# Load model
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
observations = sim_results["observations"]
hrnn = sim_results["activities_rnn"]
actions = sim_results["actions"]
actions = dy.transform_actions(actions) # Transform actions!

# pca
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=3.pkl")
pca = pca_dic["pca"]
y_pca = pca.transform(hrnn)
x1, x2, x3 = y_pca.T
min1, max1 = min(x1) - 1, max(x1) + 1
min2, max2 = min(x2) - 1, max(x2) + 1
min3, max3 = min(x3) - 1, max(x3) + 1

# plot in time
angles = list(np.ones(len(observations) - start)*29)
count = 0
for t in range(start, len(observations)):
    # Obtain the fixed points
    x_0 = observations[t]
    args = [x_0, rnn, inn, br, bi]
    fps = dy.get_fixed_points(*args)

    # get color
    vx, vy, C = observations.T
    r, theta = actions.T
    color = "b" if C[t] > 0 else "k"
    color_wind = "g" if np.arctan2(vy[t], vx[t]) > 0 else "r"
    color_agent = "g" if theta[t] > 0 else "r"

    # plot fixed points and trajectory
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 3)
    ax = fig.add_subplot(gs[:, :2], projection="3d")
    fps_pca = pca.transform(fps)
    ax.scatter(*fps_pca.T, color=color)
    ax.set_xlim(min1, max1); ax.set_ylim(min2, max2); ax.set_zlim(min3, max3)
    vis.plot_trajectory(y_pca.T[:,:t], save=False, ax=ax)

    # plot wind direction and actions
    ax2 = fig.add_subplot(gs[0, -1])
    ax2.set_facecolor(color_wind)
    ax3 = fig.add_subplot(gs[1, -1])
    ax3.set_facecolor(color_agent)
    ax2.set_title("Wind Angle"); ax3.set_title("Agent Angle")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3.set_xticks([]); ax3.set_yticks([])

    # save
    ax.view_init(30, angles[t - start])
    if not os.path.exists(os.path.join(graphpath, foldername)): os.mkdir(os.path.join(graphpath, foldername))
    vis.savefig(figname=f"{foldername}/{count}.png", close=True) # round(angles[t - start],0)
    count += 1

vis.gen_gif(False, foldername, None, stall=10, angles=list(range(len(angles))))