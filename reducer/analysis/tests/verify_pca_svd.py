import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
from sklearn.decomposition import PCA

# load model
specify = 0
tpe = "constant"
rnn, inn, br, bi = bcs.model_loader(specify=specify)
r = 3

# append trajectories
X = []
for episode in range(10):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    hrnn = dic["activities_rnn"]
    X.append(hrnn)
X = np.vstack(X)

# pca
pca = PCA(n_components=r)
y_pca = pca.fit_transform(X)

# svd
Xc = X - np.mean(X, axis=0)
U, s, Vt = np.linalg.svd(Xc)
S = np.zeros((U.shape[0], 64))
np.fill_diagonal(S, s)
y_svd = U[:, :r] @ S[:r, :r]
