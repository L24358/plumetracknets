import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from pysindy import SINDy
from sklearn.decomposition import PCA
from reducer.config import graphpath, modelpath

# load agent
specify = 0
rnn, inn, br, bi = bcs.model_loader(specify=specify)

# get all trajectories and observations
trajs = []
obs = []
for i in range(240):
    sim_results = bcs.simulation_loader(specify, "constant", episode=i)
    observations = sim_results["observations"]
    h_0 = sim_results["activities_rnn"][0]
    t, y_rnn, actions_rnn = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, specify, T=len(observations))

    trajs.append(y_rnn)
    obs.append(observations)
trajs = np.vstack(trajs)
obs = np.vstack(obs)

# perform PCA on trajectories
pca = PCA(n_components=3)
y_pca = pca.fit_transform(trajs)

# define and fit model
model = SINDy()
model.fit(trajs, u=observations)
eqs = model.equations()