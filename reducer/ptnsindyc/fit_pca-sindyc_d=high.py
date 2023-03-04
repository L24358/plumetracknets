'''
Perform SINDYc on PCA version (>3 dimensions) of PTN model hidden activity.
'''

import os
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
pca_dim = 5
clip = False
same_ic = True
thre = 1e-07
T = 1000
seed = np.random.randint(0, high=99999) 

# set np seed
np.random.seed(seed) # decent values: 43287, 35190, 91475
print("seed: ", seed)

# load agent
rnn, inn, br, bi = bcs.model_loader(specify=specify)
fit_dic = bcs.fit_loader(specify, episode)

# modify fit_dic
fit_dic["C"][0][-2:] = [0, 0]
fit_dic["y"][0][-2:] = [0, 0]
fit_dic["x"][0][-2:] = [0, 0]

# get observations
observations = bcs.FitGenerator(fit_dic).generate(np.arange(T))
observations += np.random.normal(0, 0, size=observations.shape)
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

# perform PCA on trajectories
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

# perform pca on simulation and sindy results
pca2 = PCA(n_components=3)
y_pca2 = pca.fit_transform(y_pca)
sol2 = pca.transform(sol)

# plot results
fig = plt.figure(figsize=(7,3))
ax2 = fig.add_subplot(121, projection="3d")
vis.plot_PCA_3d(y_pca2, save=False, plot_time=True, ax=ax2, subtitle="Original simulation")
ax1 = fig.add_subplot(122, projection="3d")
vis.plot_PCA_3d(sol2[100:], save=False, plot_time=True, ax=ax1, subtitle="SINDy simulation")
plt.suptitle(f"SINDy Results (seed = {seed})")
vis.savefig(figname=f"temp.png") # pca_sindyc_seed={seed}

# print model
model.print()