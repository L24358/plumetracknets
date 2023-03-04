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
to_save = {"observations": {}, "hrnns": {}, "actions": {}}
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    observations = dic["observations"]
    hrnns = dic["activities_rnn"]
    actions = dic["actions"]
    for t in range(len(actions)):
        to_save["observations"][(episode, t)] = observations[t]
        to_save["hrnns"][(episode, t)] = hrnns[t]
        to_save["actions"][(episode, t)] = actions[t]

bcs.pklsave(to_save, "pca_frame", f"stacked_agent={specify+1}_format=2.pkl")
