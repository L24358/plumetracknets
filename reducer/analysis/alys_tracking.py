import os 
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters and model loading
specify = 0
tpe = "constant"
episode = 38
start = 30
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
dic = bcs.simulation_loader(specify, tpe, episode=episode, verbose=False)

x, y, C = dic["observations"][start:].T
r, theta = dic["actions"][start:].T
hs = dic["activities_rnn"][start:]
# get_fixed_points(x, rnn, inn, br, bi, rp=100)
wind_angle = np.array([np.arctan2(x[t], y[t]) for t in range(len(x))])
odor_mask = C > 0
obs_act = np.vstack([wind_angle, odor_mask, theta]).T
print(obs_act)
