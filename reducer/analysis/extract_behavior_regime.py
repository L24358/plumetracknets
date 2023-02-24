"""
Identify behavior regime based on actions and observations.
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
c1, c2 = 12, 25

regimes = {} # key = episode, val = {"tracking", ...}
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    C = np.clip(dic["observations"].T[-1], 0, 1000) # clip off those less than 0
    filt5 = np.append(np.zeros(1), np.ones(c1))
    history5 = np.convolve(C, filt5)[:len(C)]
    filt12 = np.append(np.zeros(1), np.ones(c2))
    history12 = np.convolve(C, filt12)[:len(C)]

    tracking = (history5 > 1) # if odor is sensed in the last 5 (c1) time steps
    lost = (history12 <= 0) # if odor is not sensed in the last 12 (c2) time steps
    recovery = (1 - tracking) + (1 - lost) # not tracking, and not lost
    recovery = (recovery == 2)

    regimes[episode] = {"tracking": tracking, "recovery": recovery, "lost": lost}

bcs.pklsave(regimes, "regimes", f"regime_agent={specify+1}_criterion=({c1},{c2}).pkl")