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
rnn, inn, br, bi = bcs.model_loader(specify=specify)

# append trajectories
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    h_0 = dic["activities_rnn"][-1] # choose the last value
    obs = dic["observations"][-1]
    _, hrnn = dy.sim(rnn, inn, br, bi, dy.constant_obs(obs), h_0, T=50)

    pca = PCA(n_components=2)
    y_pca = pca.fit_transform(hrnn)
    vis.plot_trajectory_2d(y_pca.T, figname=bcs.fjoin("temp", f"{episode}.png", tpe=2))