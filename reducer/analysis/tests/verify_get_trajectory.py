import os 
import numpy as np
import reducer.support.basics as bcs
import reducer.support.visualization as vis

# hyperparameters
T = 1000

# load data
r = np.ones(T)
theta = np.ones(T)*3
actions = np.vstack([r, theta]).T
observations = np.zeros((T, 3))
vis.plot_obs_act_traj(actions, observations, figname="temp.png")

