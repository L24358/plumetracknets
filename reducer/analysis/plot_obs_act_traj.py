"""
Plot observations, actions and trajectory together for comparison.

Conclusion:
    - "Track" seems to consist of oscillatory obs and actions, e.g. episode 65 
    - "Lost" seems to consist of constant obs and high-freq oscillatory actions, e.g. episode 26, 213
    - "Recovery" -- unclear, e.g. episode 228
"""

import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

specify = 0
for episode in range(240):

    sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
    observations = sim_results["observations"]
    actions = sim_results["actions"]

    vis.plot_obs_act_traj2(actions, observations, figname=f"obs-act-traj/obs-act-traj_agent={specify+1}_tpe=constant_episode={episode}.png")