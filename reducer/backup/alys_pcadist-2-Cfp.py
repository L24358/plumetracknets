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
save = "all"
n = 3 if save == "pca" else 64
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n={n}.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]
episodes = bcs.track_dic_manual

pos = []
neg = []
pos_dic = {}
neg_dic = {}

i = 0
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    actions = dic["actions"]
    observations = dic["observations"]
    vx, vy, C = observations.T

    pos_dic[episode] = []
    neg_dic[episode] = []
    for t in range(0, len(actions)):

        x_0 = observations[t]
        args = [x_0, rnn, inn, br, bi]
        fps = dy.get_fixed_points(*args)
        fps_pca = pca.transform(fps)
        if C[t] > 0:
            pos += list(fps_pca)
            pos_dic[episode] += list(fps_pca)
        else:
            neg += list(fps_pca)
            neg_dic[episode] += list(fps_pca)
        i += 1

pos = np.array(pos).T
neg = np.array(neg).T
ax = plt.figure().add_subplot(projection="3d")
ax.scatter(*pos, color="b", s=1)
ax.scatter(*neg, color="k", s=1)
vis.gen_gif(True, "pcadist_Cfp", ax, stall=5, angle1=30, angles=None)

# save
bcs.pklsave(pos_dic, "pcadist", f"Cfp_pos_agent={specify+1}_save={save}.pkl")
bcs.pklsave(neg_dic, "pcadist", f"Cfp_neg_agent={specify+1}_save={save}.pkl")