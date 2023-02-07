'''
Plot fixed points in terms of the instantaneous inputs.
'''

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA
from reducer.support.basics import constant, single_sine
from reducer.config import modelpath

# parameters
simulate_fp = False
use_alltrajs = False
use_simulation = True
specify = 0
episode = 5

# Load model
rnn, inn, br, bi = bcs.model_loader(specify=specify) 

# Load and plot trajectories
if use_simulation: # Use artificial input and simulated results
    sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
    h_0 = sim_results["activities_rnn"][0]

    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f: dic = pickle.load(f)
    T = np.arange(100)
    C = dic["C"][1](T, *dic["C"][0])
    y = dic["y"][1](T, *dic["y"][0])
    x = dic["x"][1](T, *dic["x"][0])
    observations = np.vstack((C, y, x)).T
    _, trajs = dy.sim(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, T=100)
else: # use real input and results
    trajs = []
    sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
    observations = sim_results["observations"]
    trajs = sim_results["activities_rnn"]

if use_alltrajs: # plot all trajectories as reference
    trajs = np.load(os.path.join(modelpath, "activities_rnn", f"alltrajs_agent={specify+1}.npy"))

# Perform PCA on trajectories
pca = PCA(n_components=3)
y_pca = pca.fit_transform(trajs)
ax = plt.figure().add_subplot(projection="3d")
vis.plot_trajectory(y_pca.T, save=False, ax=ax, projection="3d")

color = sns.color_palette("viridis", len(observations))
for t in range(len(observations)):
    # Obtain the fixed points
    x_0 = observations[t]
    args = [x_0, rnn, inn, br, bi]
    fps = dy.get_fixed_points(*args)
    Js = [dy.jacobian(fp, args) for fp in fps]

    # Check how accurate the fixed point solutions are, and stability
    fps_pca = pca.transform(fps)
    for i in range(len(fps)):
        ax.scatter(*fps_pca[i], color=color[t])

        # get largest eigenvector
        # evs, ews = dy.get_sorted_eig(fps[i], args)
        # ew_max = ews[0]
        # ew_pca = pca.transform(ew_max)
        # ax.quiver(*fps_pca[i], *ew_pca, color[t])

    # Simulate and plot trial with fixed point as i.c.
    if simulate_fp:
        h_0 = fps[0] # Take the first fixed point
        t, y = dy.sim(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, T=1000)
        vis.plot_PCA_3d(y, figname="fixed_point_evolution.png") # Plot first 3 PCA dimensions

vis.gen_gif(True, f"fixed_point_episode={episode}", ax, stall=5, angle1=45)