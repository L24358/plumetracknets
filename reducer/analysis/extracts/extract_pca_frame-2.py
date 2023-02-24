import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 2
tpe = "constant"
n = 64
seed = 42

# append trajectories
X = []
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    hrnn = dic["activities_rnn"]
    X.append(hrnn)
X2 = np.vstack(X)

# pca from sklearn
pca = PCA(n_components=n, random_state=seed)
all_traj = pca.fit_transform(X2)
all_traj_sep = [pca.transform(X[i]) for i in range(240)]
dic = {"pca": pca, "all_traj": all_traj, "all_traj_sep": all_traj_sep, "all_traj_ori": X2, "all_traj_ori_sep": X}
bcs.pklsave(dic, "pca_frame", f"pcaskl_agent={specify+1}_n={n}_seed={seed}.pkl")