"""
@ Conclusion:
    - hidden activities in RNN and MLP are same as logged when using original observation
    - actions = actions_ori[:-1], and is different from actions_new, where
        actions are logged; actions_ori are simulated without transform_observation; actions_new are with
    - observations do not need to be transformed; actions do
"""

import os 
import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
episode = 224 # 66

# load data
dic = bcs.simulation_loader(specify, tpe, episode=episode)
rnn, inn, br, bi = bcs.model_loader(specify)

# fake data
T, _ = dic["observations"].shape
if 0:
    r = np.random.uniform(0., 1., T)
    theta = np.ones(T)*(-1) # positive: CCW, negative: CW
    dic["actions"] = np.vstack([r, theta]).T

# simulate using logged hs
hs = dic["activities_rnn"]
actions = dic["actions"]
observations = dic["observations"]
observations_new = dy.transform_observations(observations)
_, hs_ori, actions_ori = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(observations), hs[0], specify, T=T)
_, hs_new, actions_new = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(observations_new), hs[0], specify, T=T)

# plot
dic["actions"] = actions_ori
if 1: vis.plot_obs_act_traj2(dic["actions"], dic["observations"], figname="temp.png")
