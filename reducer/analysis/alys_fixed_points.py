import os
import pickle
import numpy as np
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
episode = 82

# Load model
rnn, inn, br, bi = bcs.model_loader(specify=specify) # Take the first model

# Load and plot trajectories
if use_simulation:
    sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
    h_0 = sim_results["activities_rnn"][0]

    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f: dic = pickle.load(f)
    T = np.arange(100)
    C = dic["C"][1](T, *dic["C"][0])
    y = dic["y"][1](T, *dic["y"][0])
    x = dic["x"][1](T, *dic["x"][0])
    observations = np.vstack((C, y, x)).T
    _, trajs = dy.sim(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, T=100)
else:
    trajs = []
    sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
    observations = sim_results["observations"]
    trajs = sim_results["activities_rnn"]

if use_alltrajs:
    trajs = np.load(os.path.join(modelpath, "activities_rnn", f"alltrajs_agent={specify+1}.npy"))

# Perform PCA
pca = PCA(n_components=3)
y_pca = pca.fit_transform(trajs)
ax = plt.figure().add_subplot(projection="3d")
ax.plot(y_pca[:,0], y_pca[:,1], y_pca[:,2], "k", alpha=0.5)

for t in range(len(observations)):
    # Obtain the fixed points
    x_0 = observations[t]
    args = [x_0, rnn, inn, br, bi]
    fps = dy.get_fixed_points(*args)
    Js = [dy.jacobian(fp, args) for fp in fps]

    # Check how accurate the fixed point solutions are, and stability
    fps_pca = pca.transform(fps)
    for i in range(len(fps)):
        print(f"sum(rhs) for fp{i}: ", sum(dy.rhs(fps[i], args)))
        evs = np.linalg.eigvals(Js[i])
        stable = np.all(abs(evs) < 1)
        print(f"Is the fixed point stable? ", stable)

        color = "g" if stable else "r"
        ax.scatter(*fps_pca[i], color=color)

    # Simulate and plot trial with fixed point as i.c.
    if simulate_fp:
        h_0 = fps[0] # Take the first fixed point
        t, y = dy.sim(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, T=1000)
        vis.plot_PCA_3d(y, figname="fixed_point_evolution.png") # Plot first 3 PCA dimensions

vis.gen_gif(True, f"fixed_point_episode={episode}", ax, stall=5, angle1=45)