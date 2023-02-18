import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy

# load model
specify = 0
tpe = "constant"
rnn, inn, br, bi = bcs.model_loader(specify=specify)
r = 3

# append trajectories
X = []
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    hrnn = dic["activities_rnn"]
    X.append(hrnn)
X = np.vstack(X)

# svd
col_mean = np.mean(X, axis=0)
Xc = X - col_mean
U, s, Vt = np.linalg.svd(Xc)

np.save("/src/data/pca_frame/U.npy", U)
np.save("/src/data/pca_frame/s.npy", s)
np.save("/src/data/pca_frame/V.npy", Vt.T)
np.save("/src/data/pca_frame/col_mean.npy", col_mean)