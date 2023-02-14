'''
Perform SINDYc on PCA version (3 dimensions) of PTN model hidden activity.
'''

import os
import sys
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
noise_perc = 0.01
seed = np.random.randint(0, high=99999) 
save = True

# overwrite by argv: seed, noise_perc, save
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    noise_perc = float(sys.argv[2])
    save = int(sys.argv[3])

# set np seed
np.random.seed(seed) # good values: 7537, 98358, 24298, 25
print("seed: ", seed)

# load agent
rnn, inn, br, bi = bcs.model_loader(specify=specify)
fit_dic = bcs.fit_loader(specify, episode)

def get_traj_obs(fit_dic):
    observations = bcs.FitGenerator(fit_dic).generate(np.arange(T))
    noise_std = abs(np.mean(observations, axis=0))*noise_perc
    observations += np.random.normal([0,0,0], noise_std, size=observations.shape)
    if clip: observations = np.clip(observations, 0, 1) # odor concentration values clip 

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

def MSE(traj1, traj2, start=0): return np.mean(pow(traj1[start:] - traj2[start:], 2))

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

# calculate error
err_train = MSE(y_pca, sol, start=100)
err_test = MSE(new_y_pca, new_sol, start=100)
row = f"seed={seed} nperc={noise_perc} trainerr={err_train} testerr={err_test}"
bcs.write_row(f"agent={specify+1}_episode={episode}_d=3.txt", "ptnsindyc_model_selection", row)
print(f"TRAIN error: {err_train}, TEST error: {err_test}")

# plot results
if save: 
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
    figname = f"pca_sindyc_agent={specify+1}_episode={episode}_seed={seed}_nperc={noise_perc}.png"
    vis.savefig(figname=figname)