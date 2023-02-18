import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

specify = 0
tpe = "constant"
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=3.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]

pos_theta = []
neg_theta = []
i = 0
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    actions = dic["actions"]
    actions = dy.transform_actions(actions)
    observations = dic["observations"]
    vx, vy, C = observations.T

    for t in range(len(actions)):
        r, theta = actions[t]
        if C[t] > 0: # only when tracking (sorta)
            if theta > 0: pos_theta.append(all_traj[i])
            else: neg_theta.append(all_traj[i])
        i += 1

pos_Cfp = []
neg_Cfp = []
for episode in [38, 16, 21, 91]:
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    observations = dic["observations"]
    vx, vy, C = observations.T

    for t in range(len(observations)):
        x_0 = observations[t]
        args = [x_0, rnn, inn, br, bi]
        fps = dy.get_fixed_points(*args)
        fps_pca = pca.transform(fps)

        if C[t] > 0:
            pos_Cfp += list(fps_pca)
        else:
            neg_Cfp += list(fps_pca)

pos_theta = np.array(pos_theta).T
neg_theta = np.array(neg_theta).T
pos_Cfp = np.array(pos_Cfp).T
neg_Cfp = np.array(neg_Cfp).T
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(*pos_theta, color="g", s=1)
ax.scatter(*neg_theta, color="r", s=1)
ax.scatter(*pos_Cfp, color="b", s=3)
ax.scatter(*neg_Cfp, color="k", s=3)
vis.gen_gif(True, "pca_dist", ax, stall=5, angle1=30, angles=None)