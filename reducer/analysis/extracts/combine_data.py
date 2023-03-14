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

# append trajectories
obs, hrnns, actions, ts = [], [], [], []
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    obs.append(dic["observations"])
    hrnns.append(dic["activities_rnn"])
    actions.append(dic["actions"])
    ts.append(np.arange(len(dic["activities_rnn"])).reshape(1,-1))
obs_stack = np.vstack(obs)
hrnns_stack = np.vstack(hrnns)
actions_stack = np.vstack(actions)
ts_stack = np.hstack(ts)

to_save = {"observations": obs_stack, "hrnns": hrnns_stack, "actions": actions_stack, "ts": ts_stack.squeeze(),
            "observations_sep": obs, "hrnns_sep": hrnns, "actions_sep": actions, "ts_sep": ts}
bcs.pklsave(to_save, "pca_frame", f"stacked_agent={specify+1}.pkl")
