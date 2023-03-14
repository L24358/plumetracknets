"""
Plot centerline versus distribution of agent angle (for both real and fixed points), DEPRECATED.
"""

import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# define class, functions for convenience (repeats from alys_ahstar)
def get_closest_stable_fp(fps, h, args):
    if len(fps) > 1: # if there are multiple fps to choose from, return closest fp and flag=False
        stable = [dy.get_stability(fp, args) for fp in fps]
        fps = [fps[i] for i in range(len(fps)) if stable[i]]
        diffs = [np.linalg.norm(fp - h) for fp in fps]
        idx = diffs.index(min(diffs))
        return fps[idx], False
    elif len(fps) == 1: # if there is only one, return only fp and flag=is_stable
        stable = dy.get_stability(fps[0], args)
        if not stable: print("Warning: only single unstable fixed point exists!")
        return fps[0], not stable
    else: # if there are no fixed points, return None and flag=True
        print("Warning: No fixed point is found!")
        return None, True    

def get_fixed_point_wrap(observations, t, args, hs, count=0, rp=100):
    x = observations[t]
    args = [x] + args
    fps = dy.get_fixed_points(*args, rp=rp)
    fp, flag = get_closest_stable_fp(fps, hs[t], args)

    if not flag: return fp # nothing is wrong
    else:
        if type(fp) == np.ndarray: return None # unstable fixed point
        if count > 3:
            print("Max iter reached!")
            return fp # if iter > max_iter, return no matter what
        else: return get_fixed_point_wrap(observations, t, args, hs, count=count+1, rp=500) # try again

def get_fixed_point_action(observations, actions, hs, args): # get fixed point actions, a(h_t^*)
    h_sequence = []
    fp_sequence = []
    ego_wind_angles = []
    abs_wind_angles = []
    for t in range(len(actions)):
        # get h_t^* and a(h_t^*)
        flag = False
        fp = get_fixed_point_wrap(observations, t, args, hs)
        if type(fp) == np.ndarray: fp_sequence.append(fp)
        else:
            flag = True
            print("Fixed point is unstable, or no fixed point is found after several attempts.")
        
        # get hs
        if not flag: h_sequence.append(hs[t])

        # get egocentric wind
        if not flag:
            x, y, C = observations[t]
            ego_wind_angle = np.arctan2(y, x)
            ego_wind_angles.append(ego_wind_angle)

    assert len(h_sequence) == len(fp_sequence) == len(ego_wind_angles)
    return h_sequence, fp_sequence, ego_wind_angles

# plot absolute agent angle (both real and instantaneous) against plume 
def get_agent_history_for_fp(actions_real, actions_star):
    agent_real = bcs.Agent()
    agent_star = bcs.Agent()

    assert len(actions_real) == len(actions_star)
    for t in range(len(actions_real)):
        # at time t...
        agent_real.update(*actions_real[t]) # not logged yet
        agent_star.set_angle(agent_real.history[-1][-1]) # set angle to the same value as agent_real
        _, abs_angle_star = agent_star.update(*actions_star[t])

        # log the new position at time t+1...
        agent_real.log()
        agent_star.log(overwrite=[None, None, abs_angle_star]) # only the angle matters (and makes sense)
    return agent_real, agent_star

def get_abs_wind_angle(agent, ego_wind_angles):
    abs_wind_angles = []
    for t in range(len(ego_wind_angles)):
        abs_wind_angle = agent.history[t][-1] + ego_wind_angles[t]
        abs_wind_angles.append(abs_wind_angle)
    return abs_wind_angles

def mask_by_odor(mask, *seqs):
    eliminate = lambda m, s: [s[t] for t in range(len(m)) if m[t] > 0]
    return [eliminate(mask, seq) for seq in seqs]

if __name__ == "__main__":
    # hyperparameters
    specify = 0
    tpe = "constant"
    episode = "random" # good trials: 128, 153

    # load data
    rnn, inn, br, bi = bcs.model_loader(specify=specify) 
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    observations = dic["observations"]
    actions = dic["actions"]
    hs = dic["activities_rnn"]
    mask = observations[:,-1] > 0

    args = [rnn, inn, br, bi]
    h_sequence, fp_sequence, ego_wind_angles = get_fixed_point_action(observations, actions, hs, args)
    actions_real = dy.get_action_from_h(specify, h_sequence, return_info=False)
    actions_star = dy.get_action_from_h(specify, fp_sequence, return_info=False)
    agent_real, agent_star = get_agent_history_for_fp(actions_real, actions_star)
    abs_wind_angles = get_abs_wind_angle(agent_real, ego_wind_angles)

    agent_real_history, agent_star_history = mask_by_odor(mask, agent_real.history[:-1,-1], agent_star.history[:-1,-1])

    counts1, _, _ = plt.hist(agent_real_history, color="b", alpha=0.5)
    counts2, _, _ = plt.hist(agent_star_history, color="r", alpha=0.5)
    maxx = max(list(counts1) + list(counts2))
    centerline = np.mean(abs_wind_angles)
    plt.plot([centerline, centerline], [0, maxx], 'k--')
    vis.savefig()
    print(f"centerline: {centerline}, real: {np.mean(agent_real_history)}, star: {np.mean(agent_star_history)}")
