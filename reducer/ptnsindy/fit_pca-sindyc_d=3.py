'''
Perform SINDYc on PCA version (3 dimensions) of PTN model hidden activity.
'''

import os
import copy
import numpy as np
import pysindy as pysindy
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
import reducer.support.sptsindy as ssy
import reducer.support.visualization as vis
from pysindy import SINDy
from scipy.integrate import odeint
from sklearn.decomposition import PCA
from reducer.config import graphpath, modelpath

# hyperparameters
specify = 0
episode = 5
pca_dim = 3
clip = False
same_ic = True
thre = 1e-07
T = 1000
noise_std = 0.0
seed = 98358 # np.random.randint(0, high=99999) 

# set np seed
np.random.seed(seed) # good values: 7537, 98358, 24298
print("seed: ", seed)

# load agent
rnn, inn, br, bi = bcs.model_loader(specify=specify)
fit_dic = bcs.fit_loader(specify, episode)

def get_traj_obs(fit_dic):
    observations = bcs.FitGenerator(fit_dic).generate(np.arange(T))
    observations += np.random.normal(0, noise_std, size=observations.shape)
    if clip: observations = np.clip(observations, 0, 1) # clip odor concentration values

    # get all trajectories and observations
    trajs, obs = [], []
    for i in range(1):
        h_0 = np.random.uniform(low=-1, high=1, size=(64,))
        t, y_rnn, actions_rnn = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, specify, T=len(observations))
        trajs.append(y_rnn[1:])
        obs.append(observations)
    trajs = np.vstack(trajs)
    obs = np.vstack(obs)
    return trajs, obs, h_0

# perform PCA on trajectories
trajs, obs, h_0 = get_traj_obs(fit_dic)
pca = PCA(n_components=pca_dim)
y_pca = pca.fit_transform(trajs)

# define and fit SINDy model
model = SINDy()
model.fit(y_pca, u=obs)

# read out fitting results
coefs = model.coefficients()
coefs[abs(coefs) < thre] = 0 # clip coefs below threshold
mask = np.where(coefs != 0, 1, 0)
cmask = np.multiply(coefs, mask)

# set initial values
if same_ic:
    y0 = pca.transform(h_0.reshape(1,-1))[0]
else:
    y0 = np.random.uniform(low=-1, high=1, size=(pca_dim,))

# simulate model results
t = np.arange(T)
u = lambda t: obs[int(t)]
sol = odeint(ssy.rhs, y0, t, args=(u, coefs, mask, pca_dim + 3))

# print model
model.print()

# novel stimulus
print("C params:", fit_dic["C"][0])
new_fit_dic = copy.deepcopy(fit_dic)
new_fit_dic["C"][0][1] = 0.9
new_fit_dic["C"][0][1] = 0.35
new_fit_dic["C"][0][2] = 0
new_trajs, new_obs, _ = get_traj_obs(new_fit_dic)

# apply new observations
pca3 = PCA(n_components=pca_dim)
new_y_pca = pca.fit_transform(new_trajs)
new_u = lambda t: new_obs[int(t)]
new_sol = odeint(ssy.rhs, y0, t, args=(new_u, coefs, mask, pca_dim + 3))

# plot results
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(221, projection="3d")
vis.plot_PCA_3d(y_pca, save=False, plot_time=True, ax=ax1, subtitle="Original simulation (training)")
ax2 = fig.add_subplot(222, projection="3d")
vis.plot_PCA_3d(sol[100:], save=False, plot_time=True, ax=ax2, subtitle="SINDy simulation (training)")
ax3 = fig.add_subplot(223, projection="3d")
vis.plot_PCA_3d(new_y_pca, save=False, plot_time=True, ax=ax3, subtitle="Original simulation (test)")
ax4 = fig.add_subplot(224, projection="3d")
vis.plot_PCA_3d(new_sol[100:], save=False, plot_time=True, ax=ax4, subtitle="SINDy simulation (test)")
plt.suptitle(f"SINDy Results (seed = {seed})")
vis.savefig(figname=f"pca_sindyc_seed={seed}.png")