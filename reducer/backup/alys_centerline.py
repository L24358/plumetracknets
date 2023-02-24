import os 
import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
episode = 5

# load data
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
dic = bcs.simulation_loader(specify, tpe, episode=episode)
observations, actions, y_rnn = [dic[key] for key in ["observations", "actions", "activities_rnn"]]
observations = dy.transform_observations(observations)
actions = dy.transform_actions(actions)

# load actor
actor_pms = bcs.actor_loader(specify=specify)
actor = dy.Actor()
actor.init_params(actor_pms)

# get action by hidden units activity
def get_action_by_h(h):
    h = np.array([h])
    actions = actor(h).detach().numpy()
    return dy.transform_actions(actions).squeeze()

# get centerline
# The "absolute coordinate" is the egocentric frame at t=0
abs_wind_theta = [] 
for t in range(len(observations)):
    x, y, _ = observations[t]
    ego_wind_theta = np.arctan2(y/x)
    agent_theta = actions[t]
    

# main
for t in range(len(observations)):
    fps = dy.get_fixed_points(observations[t], rnn, inn, br, bi)
    y = y_rnn[t]

    action_fps = [get_action_by_h(fp) for fp in fps]
    action_current = get_action_by_h(y)
    

    
