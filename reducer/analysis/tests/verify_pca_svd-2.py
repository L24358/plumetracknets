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
T1, T2 = 200, 200
C_max = 1

# load model
rnn, inn, br, bi = bcs.model_loader(specify=specify)

# construct stimulus and simulate
x = np.ones(T1 + T2)*0
y = np.ones(T1 + T2)*(-1)
C = np.append( np.ones(T1)*C_max, np.zeros(T2) )
u = np.vstack([x, y, C]).T
u += np.random.normal(0, 0, size=u.shape)
h_0 = np.random.uniform(-1, 1, size=(64,))
t, hrnn, actions = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(u), h_0, specify, T=T1+T2)

pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]
import pdb; pdb.set_trace()

# # append trajectories
# X = []
# for episode in range(240):
#     dic = bcs.simulation_loader(specify, tpe, episode=episode)
#     hrnn = dic["activities_rnn"]
#     X.append(hrnn)
# X = np.vstack(X)

# # pca from sklearn
# pca = PCA(n_components=3)
# all_traj = pca.fit_transform(X)
y_pca = pca.transform(hrnn)
ax = vis.plot_PCA_3d(None, y_pca=all_traj, plot_time=False, save=False)
vis.plot_PCA_3d(None, y_pca=y_pca, ax=ax)

# # pca from svd
# pca_svd = bcs.PCASVD(3)
# y_pca_svd = pca_svd.transform(hrnn)
# all_traj2 = pca_svd.get_base_traj()
# ax = vis.plot_PCA_3d(None, y_pca=all_traj2, plot_time=False, save=False)
# vis.plot_PCA_3d(None, y_pca=y_pca_svd, ax=ax, figname="temp2.png")

actions = dy.transform_actions(actions)
print(actions[190:200])
print(actions[-10:])
