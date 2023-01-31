'''
Fit trials with short term Fourier transform.
'''

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from scipy.signal import stft
from scipy.optimize import differential_evolution
from reducer.config import modelpath

specify = 0
episode = int(sys.argv[1])

sim_results = bcs.simulation_loader(specify, "constant", episode=episode)
observations = sim_results["observations"]
actions = sim_results["actions"]
targets = np.hstack((observations, actions)).T

def normalize(x): return (x - np.mean(x))/np.std(x)

def flat_regions(x, thre=0.1):
    regions = []
    region = []
    for i in range(len(x)-1):
        in_region = True
        if abs(x[i] - x[i+1]) > thre: in_region = False
        if in_region:
            region.append(i)
            region.append(i+1)
        elif region != []:
            regions.append(region)
            region = []
    if region != []: regions.append(region)
    return regions

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

def eval_fit(target, func):
    res = differential_evolution(MSE, bounds(func), args=(target, func))
    error = MSE(res.x, target, func)
    return res.x, error

norm = normalize(targets[-1])
f, t, Zxx = stft(norm, nperseg=20)
max_freq = np.max(np.abs(Zxx), axis=0)
max_loc = np.argmax(np.abs(Zxx), axis=0)

if 0: # plot the results of the stft
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(311)
    ax1.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    ax2 = fig.add_subplot(312)
    ax2.plot(max_freq, "k")
    ax3 = fig.add_subplot(313)
    ax3.plot(norm)
    vis.savefig(clear=True)

c = 0
regions = flat_regions(max_freq, thre=0.2)
grid = np.linspace(0, len(norm)-1, len(max_freq))
quantities = ["C", "y", "x", "r", "theta"]
for region in regions:
    if len(region) < 3: pass
    else:
        print(region)
        start, end = region[0], region[-1]
        mean_locs = np.mean(max_loc[start: end])
        if mean_locs > 0.5: funcs = [single_sine, single_sine, single_sine, single_sine, single_sine]
        else: funcs = [constant, constant, constant, constant, constant]

        start, end = int(grid[start]), int(grid[end])
        popts = {}
        trajs = [[[], [], []], [[], []]] # shape = (2, 3)
        for i in range(5):
            snippet = targets[i][start:end]
            popt, err = eval_fit(snippet, funcs[i])
            popts[quantities[i]] = popt
            idx1, idx2 = np.unravel_index(i, (2, 3))
            trajs[idx1][idx2] = [snippet, funcs[i](np.arange(len(snippet)), *popt)]

        subtitle = ["C", "y", "x", "r", "\u03B8"]
        color = ["k", "r", "b", "g", "m"]
        title = "Fit Observations, Actions"
        vis.plot_multiple_quantities2(trajs, figname=f"fit_agent={specify+1}_episode={episode}_region={c}.png", override_N=5, subtitle=subtitle, color=color, title=title)

        with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_region={c}.pkl"), "wb") as f:
            pickle.dump(popts, f)
        c += 1