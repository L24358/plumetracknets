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
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n={n}.pkl")
pca = pca_dic["pca"]
all_traj = pca_dic["all_traj"]

pos = []
neg = []
pos_thetas = []
neg_thetas = []
pos_dic = {}
neg_dic = {}
i = 0
for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    actions = dic["actions"]
    actions = dy.transform_actions(actions)
    observations = dic["observations"]
    vx, vy, C = observations.T

    pos_dic[episode] = []
    neg_dic[episode] = []
    for t in range(len(actions)):
        r, theta = actions[t]
        if C[t] > 0: # only when tracking (sorta)
            if theta > 0:
                pos.append(all_traj[i])
                pos_dic[episode].append(all_traj[i])
                pos_thetas.append(abs(theta))
            elif theta < 0:
                neg.append(all_traj[i])
                neg_dic[episode].append(all_traj[i])
                neg_thetas.append(abs(theta))
        i += 1

if save == "pca":
    assert i == len(all_traj)
    pos = np.array(pos).T
    neg = np.array(neg).T
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(*pos, c=pos_thetas, cmap="Reds", s=1)
    ax.scatter(*neg, c=neg_thetas, cmap="Greens", s=1)
    vis.gen_gif(True, "pcadist_theta", ax, stall=5, angle1=30, angles=None)

# save
bcs.pklsave(pos_dic, "pcadist", f"theta_pos_agent={specify+1}_save={save}.pkl")
bcs.pklsave(neg_dic, "pcadist", f"theta_neg_agent={specify+1}_save={save}.pkl")
bcs.npsave(np.array(pos_thetas), "pcadist", f"thetaval_pos_agent={specify+1}_save={save}.npy")
bcs.npsave(np.array(neg_thetas), "pcadist", f"thetaval_neg_agent={specify+1}_save={save}.npy")