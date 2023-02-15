"""
Perform ahstar.py, ahstar-2.py, except pooled over all trials
"""
import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from alys_ahstar import get_fixed_point_action

# hyperparameters
specify = 0
tpe = "constant"

# main loop for alys-1
instants, noninstants = [], []
for episode in range(240):
    print(episode)

    # load data
    rnn, inn, br, bi = bcs.model_loader(specify=specify) 
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    observations = dic["observations"]
    actions = dic["actions"]

    # get egocentric and absolute wind direction, hs, and a(h)s
    h_sequence, fp_sequence, ego_wind_angles = get_fixed_point_action(observations, actions)
    actions_star = dy.get_action_from_h(specify, fp_sequence, return_info=False)
    actions_real = dy.get_action_from_h(specify, h_sequence, return_info=False)

    # calculate (instant = phi - theta) and (noninstant = phi - theta*)
    instant = abs(actions_star[:,1] - ego_wind_angles)
    noninstant = abs(actions_real[:,1] - ego_wind_angles)
    instant = np.min([instant, abs(2*np.pi - instant)], axis=0)
    noninstant = np.min([noninstant, abs(2*np.pi - noninstant)], axis=0)
    instants.append(instant)
    noninstants.append(noninstant)

    # get abs agent angle
    agent_real, agent_star = get_agent_history_for_fp(actions_real, actions_star)
    abs_wind_angles = get_abs_wind_angle(agent_real, ego_wind_angles)

# plot alys-1
instants = np.array(instants).flatten()
noninstants = np.array(noninstants).flatten()
maxx = max(list(instant) + list(noninstant))
ax = plt.figure().add_subplot(111)
ax.plot([0, maxx], [0, maxx], "k--")
vis.plot_scatter(instant, noninstant, figname=f"ahstar_agent={specify+1}_all.png", xlabel="$a(h_t^*)$", ylabel="$a(h_t)$", color="b", ax=ax)