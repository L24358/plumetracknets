'''
Generate artificial stimulus using fitted parameters from a single trial.
'''

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from reducer.config import modelpath
from reducer.support.basics import constant, single_sine, combine_dict

specify = 0
episode = 5
T = 128 # Equal to batch_size
rp = 500

rnn, inn, br, bi = bcs.model_loader(specify=specify)
with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f: dic = pickle.load(f)
t = np.arange(T)
C = dic["C"][1](t, *dic["C"][0])
y = dic["y"][1](t, *dic["y"][0])
x = dic["x"][1](t, *dic["x"][0])
observations = np.vstack((C, y, x)).T

y_rnns = []
for _ in range(rp):
    h_0 = np.random.uniform(low=-1, high=1, size=(64,))
    _, y_rnn = dy.sim(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, T=T)
    y_rnns.append(y_rnn)
y_rnns = np.vstack(y_rnns)
