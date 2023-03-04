"""
Tests if recovery is biased in terms of agent angle (at the onset of recovery).
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
c1, c2 = 5, 12

# loading data
fp_pcas = bcs.npload("pcadist", f"fppca_agent={specify+1}_save=pca64.npy")
info = bcs.npload("pcadist", f"fppcainfo_agent={specify+1}_save=pca64.npy")
dic_regime = bcs.pklload("regimes", f"regime_agent={specify+1}_criterion=({c1},{c2}).pkl")

# loop
turns, turns_pos, turns_neg = [], [], []
for epi in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=epi)
    actions = dy.transform_actions(dic["actions"])
    thetas = actions.T[-1]
    filt5 = np.append(np.zeros(1), np.ones(c1))
    history5 = np.convolve(thetas, filt5)[:len(thetas)]

    prev_reg = None
    for t in range(len(thetas)):
        reg = bcs.get_regime(dic_regime, epi, t, vals=[0, 1, 2])
    
        if (prev_reg == 0) and (reg == 1):
            turns.append(np.sign(thetas[t]))

            if np.sign(history5[t]) == 1:
                turns_pos.append(np.sign(thetas[t]))
            elif np.sign(history5[t]) == -1:
                turns_neg.append(np.sign(thetas[t]))

        prev_reg = reg

print("The mean values of turns is: ", np.mean(turns))
print("The mean values of turns, conditioned on positive turn history, is: ", np.mean(turns_pos))
print("The mean values of turns, conditioned on negative turn history, is: ", np.mean(turns_neg))
print("Given that counterclockwise = +1 and clockwise = -1, is the recovery significantly biased?")