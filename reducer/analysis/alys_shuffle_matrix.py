"""
The connections of the rnn matrix seems to be randomly distributed. Would the actions change (qualitatively) when it's shuffled?
(This would be better if I can simulate the shuffled matrix performance.)

@ references:
    - https://www.tandfonline.com/doi/epdf/10.1080/13873959508837009?needAccess=true&role=button
        (Order Reduction and Determination of Dominant State Variables of Nonlinear Systems)
"""

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 0
tpe = "constant"
episode = 16

# load
rnn, inn, br, bi = bcs.model_loader(specify=specify)
dic = bcs.simulation_loader(specify, tpe, episode=episode)
obs = dic["observations"]
acts = dic["actions"]

# shuffle matrix
rnn_sffl = rnn.copy()
np.random.shuffle(rnn_sffl)
inn_sffl = inn.copy()
np.random.shuffle(inn_sffl)
_, _, acts_sffl1 = dy.sim_actor(rnn_sffl, inn, br, bi, dy.assigned_obs(obs), np.zeros(64,), specify, T=len(obs))
_, _, acts_sffl2 = dy.sim_actor(rnn, inn_sffl, br, bi, dy.assigned_obs(obs), np.zeros(64,), specify, T=len(obs))
traj = dy.get_trajectory2(acts)[0] # contains lossy transformation
traj_sffl1 = dy.get_trajectory2(acts_sffl1)[0]
traj_sffl2 = dy.get_trajectory2(acts_sffl2)[0]

# plot
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
vis.plot_trajectory_2d(traj.T, save=False, ax=ax1)
vis.plot_trajectory_2d(traj_sffl1.T, save=False, ax=ax2)
vis.plot_trajectory_2d(traj_sffl2.T, save=False, ax=ax3)
vis.savefig() # figname = f"sffl_agent={specify+1}_episode={episode}.png"