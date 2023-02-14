"""
Fit observations and actions to constant and oscillatory functions. (Manually)

@ Reason:
    - To simplify the inputs for the RNN.
"""

import os
import sys
import pickle
import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from scipy.optimize import differential_evolution
from reducer.support.basics import constant, single_sine, envelope_sine
from reducer.config import modelpath

specify = 0
episode = int(sys.argv[1])
start, end = 0, -1

sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
observations = sim_results["observations"]
actions = sim_results["actions"]
targets = np.hstack((observations, actions))[start: end]
targets = targets.T

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
    funcs = [single_sine, constant]
    for func in funcs:
        res = differential_evolution(MSE, bounds(func), args=(target, func))
        error = MSE(res.x, target, func)
        dic[error] = [res.x, func]
    err_ss, err_c = list(dic.keys())
    if err_ss < err_c/1.5: minn = err_ss # discount factor for constant
    else: minn = err_c 

    return *dic[minn], minn

popts = {}
quantities = ["C", "y", "x", "r", "theta"]
trajs = [[[], [], []], [[], []]] # shape = (2, 3)
fitfuncs = bcs.FitFuncs()
for i in range(5):
    popt, func, err = eval_fit(targets[i])
    popts[quantities[i]] = [popt, fitfuncs(func, reverse=True), err]
    idx1, idx2 = np.unravel_index(i, (2, 3))
    trajs[idx1][idx2] = [targets[i], func(np.arange(len(targets[i])), *popt)]

subtitle = ["x", "y", "C", "r", "\u03B8"]
color = ["k", "r", "b", "g", "m"]
title = "Fit Observations, Actions"
vis.plot_multiple_quantities2(trajs, figname=f"fit_agent={specify+1}_episode={episode}.png", override_N=5, subtitle=subtitle, color=color, title=title)

with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "wb") as f:
    pickle.dump(popts, f)