import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

specify = 0
x_0 = [5, 1, 2]
h_0 = np.random.normal(size=64)
rnn, inn, br, bi = bcs.model_loader(specify=specify)
t, y, actions = dy.sim_actor(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, specify, T=25)
traj = dy.get_trajectory(actions)
vis.plot_trajectory(traj.T)