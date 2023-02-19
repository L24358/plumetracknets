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
to_save = "all"

rnn, inn, br, bi = bcs.model_loader(specify=specify) 
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=3.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]
episodes = bcs.track_dic_manual

pos = []
neg = []
pos_dic = {}
neg_dic = {}

i = 0
for episode in [38, 16, 21, 91]: ## episodes.keys()
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    actions = dic["actions"]
    observations = dic["observations"]
    vx, vy, C = observations.T

    pos_dic[episode] = []
    neg_dic[episode] = []
    for t in range(0, len(actions)): ## episodes[episode]

        x_0 = observations[t]
        args = [x_0, rnn, inn, br, bi]
        fps = dy.get_fixed_points(*args)
        fps_pca = pca.transform(fps)
        angle = np.arctan2(vy[t], vx[t])
        quantity = fps_pca if to_save == "pca" else fps
        if angle > 0:
            pos += list(quantity)
            pos_dic[episode] += list(quantity)
        else:
            neg += list(quantity)
            neg_dic[episode] += list(quantity)
        i += 1

if to_save == "pca":
    pos = np.array(pos).T
    neg = np.array(neg).T
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(*pos, color="c", s=1)
    ax.scatter(*neg, color="m", s=1)
    vis.gen_gif(True, "pcadist_vfp", ax, stall=5, angle1=30, angles=None)

# save
bcs.pklsave(pos_dic, "pcadist", f"vfp_pos_agent={specify+1}_save={to_save}.pkl")
bcs.pklsave(neg_dic, "pcadist", f"vfp_neg_agent={specify+1}_save={to_save}.pkl")