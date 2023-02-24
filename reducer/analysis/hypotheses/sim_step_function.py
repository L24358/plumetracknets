import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
T1, T2 = 200, 200
C_max = 0

# load model
rnn, inn, br, bi = bcs.model_loader(specify=specify)

# construct stimulus and simulate
x = y = np.ones(T1 + T2)
C = np.append( np.ones(T1)*C_max, np.zeros(T2) )
u = np.vstack([x, y, C]).T
h_0 = np.random.uniform(-1, 1, size=(64,))
t, hrnn, actions = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(u), h_0, specify, T=T1+T2)

# PCA
pca = bcs.PCASVD(3)
y_pca = pca.transform(hrnn)
all_traj = pca.get_base_traj()
ax = vis.plot_PCA_3d(None, y_pca=all_traj, plot_time=False, save=False)
vis.plot_PCA_3d(None, y_pca = y_pca, ax=ax)