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

# eigenvalue spectrum
quantities, colors = [], []
color_green = sns.color_palette("light:b", 10) ## len(observations)
for t in range(10): # only last 10 points
    # Obtain the fixed points
    x_0 = observations[-t]
    args = [x_0, rnn, inn, br, bi]
    fps = dy.get_fixed_points(*args)
    Js = [dy.jacobian(fp, args) for fp in fps]

    # Analyze eigenvalue spectrum
    for J in Js:
        evs, ews = np.linalg.eig(J)
        evs_sorted = sorted(abs(evs), reverse=True)
        quantities.append(evs_sorted)
        colors.append(color_green[t])

vis.plot_quantities(quantities, figname="temp.png", save=True, color=colors)