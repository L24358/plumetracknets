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

pos = []
neg = []
i = 0
for episode in [38, 16, 21, 91]:
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    actions = dic["actions"]
    observations = dic["observations"]
    vx, vy, C = observations.T

    for t in range(len(actions)):
        x_0 = observations[t]
        args = [x_0, rnn, inn, br, bi]
        fps = dy.get_fixed_points(*args)
        fps_pca = pca.transform(fps)
        if C[t] > 0:
            pos += list(fps_pca)
        else:
            neg += list(fps_pca)
        i += 1

pos = np.array(pos).T
neg = np.array(neg).T
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(*pos, color="b", s=1)
ax.scatter(*neg, color="k", s=1)
vis.gen_gif(True, "Cfp_dist", ax, stall=5, angle1=30, angles=None)