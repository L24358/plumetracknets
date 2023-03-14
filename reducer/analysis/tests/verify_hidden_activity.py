"""
Verify that the logged hidden activity values are the same as the simulated ones. --> They are
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
rnn, inn, br, bi = bcs.model_loader(specify=specify)
dic = bcs.simulation_loader(specify, tpe, episode=episode)
T, _ = dic["observations"].shape

# simulate using observations
if 0: dic["observations"] = dy.transform_observations(dic["observations"])
hs = dic["activities_rnn"]
_, hs_sim, _ = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(dic["observations"]), hs[0], specify, T=T)

for t in range(T):
    print(bcs.different(hs[t], hs_sim[t], threshold=1e-4))

