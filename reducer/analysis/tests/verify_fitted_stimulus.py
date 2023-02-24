'''
Verify the artificial (fitted) stimulus.
'''

import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from reducer.support.basics import single_sine, constant

# Load model and artificial (fitted) stimulus
specify = 0
rnn, inn, br, bi = bcs.model_loader(specify=specify)
fit_params = bcs.fit_loader(specify, 65)

# Simulate
obs = dy.single_sine_obs(fit_params)
h_0 = np.random.uniform(low=-1, high=1, size=64)
t, y, actions = dy.sim_actor(rnn, inn, br, bi, obs, h_0, specify, T=125)
traj = dy.get_trajectory(actions)
vis.plot_trajectory(traj.T)