"""
Fit observations and actions to constant and oscillatory functions.

Results:
    No general way of fitting all functions. Might need to manually fit individual trials.
"""

import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from scipy.signal import stft
from scipy.optimize import differential_evolution

specify = 0
episode = 63

sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
observations = sim_results["observations"]
actions = sim_results["actions"]
targets = np.hstack((observations, actions)).T

def constant(t, b): return np.ones(len(t))*b

def single_sine(t, A, f, phi, b, s):
    return A*np.sin(f*t + phi) + b + s*t

def envelope_sine(t, A, f_slow, phi_slow, f_fast, phi_fast, b, s):
    return A*np.sin(f_slow*t + phi_slow)*np.sin(f_fast*t + phi_fast) + b + s*t

def MSE(popts, *args):
    target, func = args    
    estimate = func(np.arange(len(target)), *popts)
    return pow(target - estimate, 2).sum()
    
def bounds(func):
    if func == single_sine: return [(0, 6), (0, 1.5), (0, 2*np.pi), (-2, 3), (-2, 3)]
    elif func == envelope_sine: return [(0, 6), (0, 0.01), (0, 2*np.pi), (0.5, 1.5), (0, 2*np.pi), (-2, 3), (-2, 3)]
    elif func == constant: return [(-2, 6)]

def eval_fit(target):
    dic = {}
    funcs = [single_sine, constant, envelope_sine]
    for func in funcs:
        res = differential_evolution(MSE, bounds(func), args=(target, func))
        error = MSE(res.x, target, func)
        dic[error] = [res.x, func]
    err_ss, err_c, err_es = list(dic.keys())
    if err_es < err_ss/1.1: minn = err_es # discount factor for envelope
    elif err_ss < err_c/1.5: minn = err_ss # discount factor for constant
    else: minn = err_c 
    print(dic.keys())
    print(minn)
    return dic[minn]

popts = {}
trajs = [[[], [], []], [[], []]] # shape = (2, 3)
for i in range(5):
    popt, func = eval_fit(targets[i])
    popts[i] = popt
    idx1, idx2 = np.unravel_index(i, (2, 3))
    trajs[idx1][idx2] = [targets[i], func(np.arange(len(targets[i])), *popts[i])]

subtitle = ["C", "y", "x", "r", "\u03B8"]
color = ["k", "r", "b", "g", "m"]
title = "Fit Observations, Actions"
vis.plot_multiple_quantities2(trajs, figname="temp.png", override_N=5, subtitle=subtitle, color=color, title=title)
