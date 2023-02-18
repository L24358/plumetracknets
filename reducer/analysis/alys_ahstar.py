import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# define class, functions for convenience
def get_closest_stable_fp(fps, h, args):
    if len(fps) > 1: # if there are multiple fps to choose from, return closest fp and flag=False
        stable = [dy.get_stability(fp, args) for fp in fps]
        if np.any(stable):
            fps = [fps[i] for i in range(len(fps)) if stable[i]]
            diffs = [np.linalg.norm(fp - h) for fp in fps]
            idx = diffs.index(min(diffs))
            return fps[idx], False
        else:
            return None, True # if none of the fixed points are stable
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

def get_averaged_wind_angle(ego_wind_angles, tau):
    expfilter = lambda t, tau: np.exp(-t/tau) # Do I need to flip this?
    filter = expfilter(np.arange(len(ego_wind_angles)), tau)
    return np.convolve(ego_wind_angles, filter)[:len(ego_wind_angles)]

def get_fixed_point_action(observations, actions, hs, args): # get fixed point actions, a(h_t^*)
    h_sequence = []
    fp_sequence = []
    ego_wind_angles = []
    abs_wind_angles = []
    ts_exclude = list(range(len(actions)))
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

        # remove from tlist
        if not flag: ts_exclude.remove(t)

    assert len(h_sequence) == len(fp_sequence) == len(ego_wind_angles)
    return h_sequence, fp_sequence, ego_wind_angles, ts_exclude

if __name__ == "__main__":

    # hyperparameters
    specify = 0
    tpe = "constant"
    episode = "random"

    # load data
    rnn, inn, br, bi = bcs.model_loader(specify=specify) 
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    observations = dic["observations"]
    actions = dic["actions"]
    hs = dic["activities_rnn"]

    # get egocentric and absolute wind direction
    args = [rnn, inn, br, bi]
    h_sequence, fp_sequence, ego_wind_angles, _ = get_fixed_point_action(observations, actions, hs, args)

    # compare
    actions_star = dy.get_action_from_h(specify, fp_sequence, return_info=False)
    actions = dy.get_action_from_h(specify, h_sequence, return_info=False)
    instant = abs(actions_star[:,1] - ego_wind_angles)
    noninstant = abs(actions[:,1] - ego_wind_angles)
    instant = np.min([instant, abs(2*np.pi - instant)], axis=0)
    noninstant = np.min([noninstant, abs(2*np.pi - noninstant)], axis=0)

    maxx = max(list(instant) + list(noninstant))
    ax = plt.figure().add_subplot(111)
    ax.plot([0, maxx], [0, maxx], "k--")
    vis.plot_scatter(instant, noninstant, figname=f"ahstar_agent={specify+1}.png", xlabel="$a(h_t^*)$", ylabel="$a(h_t)$", color="b", ax=ax, s=5)

    # mutual information
    if 0:
        tau = 7
        numstate = 10
        center_wind_angles = get_averaged_wind_angle(ego_wind_angles, tau)
        fMI = bcs.time_shift_MI_wrap(instant, ego_wind_angles, 10, 1, numstate)
        print(max(fMI))
        fMI = bcs.time_shift_MI_wrap(noninstant, ego_wind_angles, 10, 1, numstate)
        print(max(fMI))
        fMI = bcs.time_shift_MI_wrap(instant, center_wind_angles, 10, 1, numstate)
        print(max(fMI))
        fMI = bcs.time_shift_MI_wrap(noninstant, center_wind_angles, 10, 1, numstate)
        print(max(fMI))