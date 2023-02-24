'''
Simulate actions of reduced matrix.
'''

import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# Load model
specify = 0
rnn, inn, br, bi = bcs.model_loader(specify=specify)

# Initial conditions
x_0 = [0, 0, 0] # concentration, y, x
h_0 = np.random.uniform(low=-1, high=1, size=64) # random initial hidden states

# SVD on rnn matrix
r = 50
rrnn = dy.low_rank_approximation(rnn, r) # "reduced" rnn
print("Is the approximated matrix different from the original? ", bcs.different(rnn, rrnn))

# Simulate the rrnn and rnn
t, y_rrnn, actions_rrnn = dy.sim_actor(rrnn, inn, br, bi, dy.constant_obs(x_0), h_0, specify, T=100)
traj_rrnn = dy.get_trajectory(actions_rrnn)
t, y_rnn, actions_rnn = dy.sim_actor(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, specify, T=100)
traj_rnn = dy.get_trajectory(actions_rnn)
fig = vis.plot_multiple_trajectory2([[traj_rnn.T, traj_rrnn.T]], save=False)
vis.common_col_title(fig, ["original", "approx"], [1, 2])
vis.savefig(figname=f"svd_constant_r={r}.png")

# Use realistic observation values
sim_results = bcs.simulation_loader(specify, "constant", episode="random")
observations = sim_results["observations"]
t, y_rrnn, actions_rrnn = dy.sim_actor(rrnn, inn, br, bi, dy.assigned_obs(observations), h_0, specify, T=len(observations))
traj_rrnn = dy.get_trajectory(actions_rrnn)
t, y_rnn, actions_rnn = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, specify, T=len(observations))
traj_rnn = dy.get_trajectory(actions_rnn)
fig = vis.plot_multiple_trajectory2([[traj_rnn.T, traj_rrnn.T]], save=False)
vis.common_col_title(fig, ["original", "approx"], [1, 2])
vis.savefig(figname=f"svd_realistic_r={r}.png")