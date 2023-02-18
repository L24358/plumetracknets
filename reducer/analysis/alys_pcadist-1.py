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
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=3.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]

pos = []
neg = []
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
            if theta > 0: pos.append(all_traj[i])
            else: neg.append(all_traj[i])
        i += 1

pos = np.array(pos).T
neg = np.array(neg).T
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(*pos, color="g", s=1)
ax.scatter(*neg, color="r", s=1)
vis.gen_gif(True, "theta_dist", ax, stall=5, angle1=30, angles=None)