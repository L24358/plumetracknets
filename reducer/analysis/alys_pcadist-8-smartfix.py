"""
Instead of re-running everything whenever I need a new quantity, why not stop being dumb and store the raw data?
Stores: pca.transform(fixed points) and its current observations + actions.
"""

import os
import pickle
import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 0
tpe = "constant"
to_save = "pca64"

# load
rnn, inn, br, bi = bcs.model_loader(specify=specify) 
pca_dic64 = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca64 = pca_dic64["pca"]
all_traj = pca_dic64["all_traj"]

fp_pcas = [] # contains the coordinates of the fixed points after pca64.transform
info = [] # contains information: (vx, vy, C, r, theta)

for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    actions = dic["actions"]
    actions = dy.transform_actions(actions) # ACTIONS are already transformed!
    observations = dic["observations"]

    for t in range(0, len(actions)):

        # get fixed points and pca transform it
        x_0 = observations[t]
        args = [x_0, rnn, inn, br, bi]
        fps = dy.get_fixed_points(*args)
        fps_pca = pca64.transform(fps)

        # append info
        fp_pcas.append(fps_pca)
        for _ in range(len(fps)): # append the same thing #fps times
            to_record = np.append(
                np.append(observations[t], actions[t]),
                np.array([episode, t]))
            info.append(to_record)

# save
fp_pcas = np.vstack(fp_pcas)
info = np.vstack(info)
bcs.npsave(fp_pcas, "pcadist", f"fppca_agent={specify+1}_save={to_save}.npy")
bcs.npsave(info, "pcadist", f"fppcainfo_agent={specify+1}_save={to_save}.npy")
