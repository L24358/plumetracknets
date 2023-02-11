import os 
import numpy as np
import reducer.support.basics as bcs
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
episode = "random"

# load data
dic = bcs.simulation_loader(specify, tpe, episode=episode)

# fake data
if 0:
    N, _ = dic["observations"].shape
    r = np.random.uniform(0, 1, size=N)*1
    theta = np.ones(N)*(1)
    dic["actions"] = np.vstack([r, theta]).T

# plot
vis.plot_obs_act_traj2(dic["actions"], dic["observations"], figname="temp.png")

