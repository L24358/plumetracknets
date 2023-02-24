import sys
sys.path.append("/src/tracer/ppo/")
sys.path.append("/src/tracer/ppo/a2c_ppo_acktr/")
import numpy as np
from gym import spaces
from model import Policy, MLPBase

if 0:
    action_space = spaces.Box(low=0, high=+1, shape=(2,), dtype=np.float32)
    mdl = Policy([100, 3], action_space, base=MLPBase, base_kwargs={
                        'recurrent': True,
                        'rnn_type': "VRNN",
                        'hidden_size': 64,
                        })

#==================================

import os
import torch
import reducer.support.navigator as nav
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# load Policy
fs = nav.file_finder()
print(fs)
# mdl = torch.load(f, map_location=torch.device('cpu'))[-1]

# hyperparameters
specify = 0
tpe = "constant"
episode = 224 # 66

# load data
dic = bcs.simulation_loader(specify, tpe, episode=episode)

import pdb; pdb.set_trace()