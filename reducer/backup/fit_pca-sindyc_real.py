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
cut = 30

# overwrite by argv: seed, noise_perc, save
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    noise_perc = float(sys.argv[2])
    save = int(sys.argv[3])

# set np seed
np.random.seed(seed) # good values: 7537, 98358, 24298, 25
print("seed: ", seed)

# load data
stackdic = bcs.pklload("pca_frame", f"stacked_agent={specify+1}.pkl")
pcadic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca = pcadic["pca"]
trajs = stackdic["hrnns_sep"]
obs = stackdic["observations_sep"]
ts = stackdic["ts_sep"]

# sort data
ts = np.arange(cut)
trajs_cut, obs_cut = [], []
for epi in bcs.track_dic_manual.keys():
    if len(trajs[epi]) >= cut:
        traj_pca = pca.transform(trajs[epi][-cut:, :])
        trajs_cut.append(traj_pca[:, :pca_dim])
        obs_cut.append(obs[epi][-cut:, :])
y_pca = np.array(trajs_cut)
obs = np.array(obs_cut)

def MSE(traj1, traj2, start=0): return np.mean(pow(traj1[start:] - traj2[start:], 2))

# define and fit SINDy model
model = SINDy()
model.fit(y_pca, u=obs, t=ts)

# read out fitting results
coefs = model.coefficients()
coefs[abs(coefs) < thre] = 0 # clip coefs below threshold
mask = np.where(coefs != 0, 1, 0)
cmask = np.multiply(coefs, mask)

# simulate model results
choice = 0
y0 = y_pca[choice][0]
u = lambda t: obs[choice][int(t)]
sol = odeint(ssy.rhs, y0, ts, args=(u, coefs, mask, pca_dim + 3))

# print model
model.print()

# novel stimulus
new_u = lambda t: obs[choice + 1][int(t)]
y0 = y_pca[choice+1][0]
new_sol = odeint(ssy.rhs, y0, ts, args=(new_u, coefs, mask, pca_dim + 3))

# calculate error
# err_train = MSE(y_pca, sol, start=100)
# err_test = MSE(new_y_pca, new_sol, start=100)
# row = f"seed={seed} nperc={noise_perc} trainerr={err_train} testerr={err_test}"
# bcs.write_row(f"agent={specify+1}_episode={episode}_d=3.txt", "ptnsindyc_model_selection", row)
# print(f"TRAIN error: {err_train}, TEST error: {err_test}")

# plot results
if save: 
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(221, projection="3d")
    vis.plot_PCA_3d(y_pca[choice][:,:3], save=False, plot_time=True, ax=ax1, subtitle="Original simulation (training)")
    ax2 = fig.add_subplot(222, projection="3d")
    vis.plot_PCA_3d(sol[:,:3], save=False, plot_time=True, ax=ax2, subtitle="SINDy simulation (training)")
    ax3 = fig.add_subplot(223, projection="3d")
    vis.plot_PCA_3d(y_pca[choice+1][:,:3], save=False, plot_time=True, ax=ax3, subtitle="Original simulation (test)")
    ax4 = fig.add_subplot(224, projection="3d")
    vis.plot_PCA_3d(new_sol[:,:3], save=False, plot_time=True, ax=ax4, subtitle="SINDy simulation (test)")
    plt.suptitle(f"SINDy Results (seed = {seed})")
    figname = f"pca_sindyc_agent={specify+1}_episode={episode}_seed={seed}_nperc={noise_perc}.png"
    vis.savefig(figname=figname)