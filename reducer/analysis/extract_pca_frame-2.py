import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 0
tpe = "constant"

# append trajectories
X = []
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    hrnn = dic["activities_rnn"]
    X.append(hrnn)
X = np.vstack(X)

# pca from sklearn
pca = PCA(n_components=3)
all_traj = pca.fit_transform(X)
dic = {"pca": pca, "all_traj": all_traj}
bcs.pklsave(dic, "pca_frame", f"pcaskl_agent={specify+1}.pkl")