'''
Plot fixed points in terms of the instantaneous inputs, evolving in time.
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
from reducer.config import modelpath, graphpath

# parameters
simulate_fp = False
use_alltrajs = False
use_simulation = False
clip = False
specify = 0
episode = 16
foldername = f"fixed_point_t_episode={episode}"

# Load model
rnn, inn, br, bi = bcs.model_loader(specify=specify) 

# Load and plot trajectories
if use_simulation: # Use artificial input and simulated results
    sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
    h_0 = sim_results["activities_rnn"][0]

    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f: dic = pickle.load(f)
    T = np.arange(100)
    fitfunc = bcs.FitFuncs()
    C = fitfunc(dic["C"][1])(T, *dic["C"][0])
    y = fitfunc(dic["y"][1])(T, *dic["y"][0])
    x = fitfunc(dic["x"][1])(T, *dic["x"][0])
    observations = np.vstack((C, y, x)).T
    _, trajs = dy.sim(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, T=100)
else: # use real input and results
    trajs = []
    sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
    observations = sim_results["observations"]
    trajs = sim_results["activities_rnn"]
if clip: observations = np.clip(observations, 0, 1) # odor concentration values clip 

if use_alltrajs: # plot all trajectories as reference
    trajs = np.load(os.path.join(modelpath, "activities_rnn", f"alltrajs_agent={specify+1}.npy"))

# Perform PCA on trajectories
pca = PCA(n_components=3)
y_pca = pca.fit_transform(trajs)

# plot in time
color = sns.color_palette("viridis", len(observations))
angles = [round(i,0) for i in np.linspace(-180, 180, len(observations))]
for t in range(len(observations)):
    # Obtain the fixed points
    x_0 = observations[t]
    args = [x_0, rnn, inn, br, bi]
    fps = dy.get_fixed_points(*args)
    Js = [dy.jacobian(fp, args) for fp in fps]

    # plot fixed points and trajectory
    ax = plt.figure().add_subplot(projection="3d")
    fps_pca = pca.transform(fps)
    ax.scatter(*fps_pca.T, color=color[t])
    vis.plot_trajectory(y_pca.T[:,:t], save=False, ax=ax, projection="3d")

    # save
    ax.view_init(30, angles[t])
    if not os.path.exists(os.path.join(graphpath, foldername)): os.mkdir(os.path.join(graphpath, foldername))
    vis.savefig(figname=f"{foldername}/{round(angles[t],0)}.png", close=True)

vis.gen_gif(False, foldername, None, stall=5, angles=angles)