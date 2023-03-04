"""
SVM on agent angle (theta).
"""

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
from sklearn.svm import LinearSVC, SVC

# functions
def concentration_pos(info):
    vx, vy, C, r, theta = info[:5] # unpack
    return True, C > 0

def phi_pos(info):
    vx, vy, C, r, theta = info[:5] # unpack
    angle = np.arctan2(vy, vx)
    return abs(angle) < np.pi/2, angle > 0

def theta_pos(action):
    r, theta = action
    return True, theta > 0

# hyperparameters
specify = 0
tpe = "constant"

# load results
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca = pca_dic["pca"]
fp_pcas = bcs.npload("pcadist", f"fppca_agent={specify+1}_save=pca64.npy")
info = bcs.npload("pcadist", f"fppcainfo_agent={specify+1}_save=pca64.npy")
hs = pca.inverse_transform(fp_pcas)
actions = dy.get_action_from_h(specify, hs, return_info=False, transform=True)

# get labels
fp_plane, fp_dots = [], []
labels_plane, labels_dots = [], []
for i in range(len(fp_pcas)):
    include, label = phi_pos(info[i])
    if include:
        fp_plane.append(fp_pcas[i])
        labels_plane.append(label)
    include, label = theta_pos(actions[i])
    if include:
        fp_dots.append(fp_pcas[i])
        labels_dots.append(label)

# perform SVM
clf = LinearSVC()
clf.fit(fp_plane, labels_plane)
print("Score: ", clf.score(fp_dots, labels_dots))