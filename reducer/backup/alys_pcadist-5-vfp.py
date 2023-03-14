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
to_save = "pca"

rnn, inn, br, bi = bcs.model_loader(specify=specify) 
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=3.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]
episodes = bcs.track_dic_manual

pos = []
neg = []
pos_vangles = []
neg_vangles = []
pos_dic = {}
neg_dic = {}

i = 0
for episode in episodes.keys(): ## episodes.keys(), [38, 16, 21, 91]
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
            pos_vangles += [abs(angle)/np.pi] * len(fps)
        else:
            neg += list(quantity)
            neg_dic[episode] += list(quantity)
            neg_vangles += [abs(angle)/np.pi] * len(fps)
        i += 1

if to_save == "pca":
    pos = np.array(pos).T
    neg = np.array(neg).T
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(*pos, c=pos_vangles, cmap="BuGn", s=1)
    ax.scatter(*neg, c=neg_vangles, cmap="PuRd", s=1)
    vis.gen_gif(True, "pcadist_vfp", ax, stall=5, angle1=15, angles=None)

# save
# bcs.pklsave(pos_dic, "pcadist", f"vfp_pos_agent={specify+1}_save={to_save}.pkl")
# bcs.pklsave(neg_dic, "pcadist", f"vfp_neg_agent={specify+1}_save={to_save}.pkl")
# bcs.npsave(np.array(pos_vangles), "pcadist", f"vangleval_pos_agent={specify+1}_save={to_save}.npy")
# bcs.npsave(np.array(neg_vangles), "pcadist", f"vangleval_neg_agent={specify+1}_save={to_save}.npy")