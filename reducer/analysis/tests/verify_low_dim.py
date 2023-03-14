'''
Verify that the model indeed generates low dimensional neuronal activity.
Plot the PCA results.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA
from reducer.config import graphpath, modelpath

specify = 0
rnn, inn, br, bi = bcs.model_loader(specify=specify)

trajs = []
for i in range(240):
    sim_results = bcs.simulation_loader(specify, "constant", episode=i)
    observations = sim_results["observations"]
    h_0 = sim_results["activities_rnn"][0]
    t, y_rnn, actions_rnn = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(observations), h_0, specify, T=len(observations))
    trajs.append(y_rnn)
trajs = np.vstack(trajs)

# save trajectories
if 0:
    np.save(os.path.join(modelpath, "activities_rnn", f"alltrajs_agent={specify+1}.npy"), trajs)
    import pdb; pdb.set_trace()

pca = PCA(n_components=3)
y_pca = pca.fit_transform(trajs)
top3 = pca.components_ # shape = (#pca dim, 64)
inn_pca = pca.transform(inn.T)
explained = pca.explained_variance_ratio_.cumsum()

ax = plt.figure().add_subplot(projection="3d")
ax.plot(y_pca[:,0], y_pca[:,1], y_pca[:,2], "k", alpha=0.5)
origin = np.mean(y_pca, axis=0)

colors = ["r", "g", "b"]
labels = ["C", "y", "x"]
for i in range(3):
    coors = np.vstack((origin, origin + inn_pca[i]*5))
    ax.plot(coors[:,0], coors[:,1], coors[:,2], colors[i], label=labels[i])

plt.legend(loc="upper left")
vis.gen_gif(True, "inn_pca", ax, stall=10, angle1=-160)