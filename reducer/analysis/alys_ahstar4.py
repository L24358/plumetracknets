"""
Perform ahstar.py, ahstar-2.py, except pooled over all trials
"""
import os 
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from alys_ahstar import get_fixed_point_action
from alys_ahstar2 import get_agent_history_for_fp, get_abs_wind_angle, mask_by_odor

# hyperparameters
specify = 0
tpe = "constant"

# functions
def adjust_mask(mask, exclude):
    for ex in exclude: mask = np.delete(mask, ex)
    return mask

def idx_by_length(l):
    lengths = [len(i) for i in l]
    return np.flip(np.argsort(lengths))

def idx_by_sum(l):
    lengths = [sum(i) for i in l]
    return np.flip(np.argsort(lengths))

# main loop for alys-1
instants, noninstants = [], []
history_real, history_star = [], []
centerlines = []
masks = []

episodes = list(range(240))
episodes.pop(178) # that one is not ran
for episode in episodes:

    dic = bcs.simulation_loader(specify, tpe, episode=episode, verbose=False)
    observations = dic["observations"]
    mask = observations[:,-1] > 0

    instant = bcs.npload("ahstar", f"instant_agent={specify+1}_episode={episode}.npy")
    noninstant = bcs.npload("ahstar", f"noninstant_agent={specify+1}_episode={episode}.npy")
    agent_real_history = bcs.npload("ahstar", f"realhistory_agent={specify+1}_episode={episode}.npy")
    agent_star_history = bcs.npload("ahstar", f"starhistory_agent={specify+1}_episode={episode}.npy")
    abs_wind_angles = bcs.npload("ahstar", f"abswindangle_agent={specify+1}_episode={episode}.npy")
    centerline = np.mean(bcs.npload("ahstar", f"abswindangle_agent={specify+1}_episode={episode}.npy"))
    
    instants += list(instant)
    noninstants += list(noninstant)
    history_real.append(agent_real_history)
    history_star.append(agent_star_history)
    centerlines.append(centerline)
    masks.append(mask)
    
# plot alys-1
instants = np.array(instants).flatten()
noninstants = np.array(noninstants).flatten()
maxx = max(list(instant) + list(noninstant))
ax = plt.figure().add_subplot(111)
ax.plot([0, maxx], [0, maxx], "k--")
sns.kdeplot(x=instants, y=noninstants, cmap="Reds", shade=True, bw_adjust=.5, cbar=True, cbar_kws={'label': 'probability density'})
ax.set_xlabel("$a(h_t^*)$"); ax.set_ylabel("$a(h_t)$")
vis.savefig(figname=f"ahstar1_agent={specify+1}_all.png")

# plot alys-2
fig = plt.figure(figsize=(3*3, 3*3))
idx = idx_by_sum(masks)[:9]
for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    counts2, _, _ = plt.hist(history_star[idx[i]], color="r", alpha=0.5, label="$a(h_t^*)$")
    counts1, _, _ = plt.hist(history_real[idx[i]], color="b", alpha=0.5, label="$a(h_t)$")
    maxx = max(list(counts1) + list(counts2))
    plt.plot([centerlines[idx[i]], centerlines[idx[i]]], [0, maxx], 'k--')
plt.legend()
vis.savefig(figname=f"ahstar2_agent={specify+1}_all.png")

# plot alys-2
mse_reals = []
mse_stars = []
for i in range(len(centerlines)):
    if len(history_real[i]) > 0:
        mse_real = pow(history_real[i] - centerlines[i], 2).mean()
        mse_star = pow(history_star[i] - centerlines[i], 2).mean()
        mse_reals.append(mse_real)
        mse_stars.append(mse_star)
maxx = max(mse_reals + mse_stars)
ax = plt.figure(figsize=(4,4)).add_subplot(111)
ax.scatter(mse_stars, mse_reals, color="b")
ax.plot([0, maxx], [0, maxx], "k--")
ax.set_xlabel("$a(h_t^*)$"); ax.set_ylabel("$a(h_t)$")
vis.savefig(figname=f"ahstar2-2_agent={specify+1}_all.png")
