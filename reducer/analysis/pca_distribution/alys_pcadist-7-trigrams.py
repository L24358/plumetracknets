"""
Label the trajectory in terms of the 8 factions or 4 quadrents, and find the trigrams.
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
from itertools import product

# hypoerparameters
specify = 0

# load results
pca_dic64 = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca64 = pca_dic64["pca"]
all_traj_sep = pca_dic64["all_traj_sep"]
svc_dic = bcs.pklload("pcadist", f"svc_agent={specify+1}.pkl")
clf1, clf2, clf3 = svc_dic.values()

fac = {}
prod = list(product((0,1),(0,1),(0,1)))
for i in range(8): fac[prod[i]] = 8-i
quad = {(1,1): 1, (0,1): 2, (0,0): 3, (1,0):4}

trigrams = bcs.get_grams([], [1,2,3,4], n=3)
episodes = bcs.track_dic_manual.keys()
for episode in episodes:
    start = bcs.track_dic_manual[episode]
    traj = all_traj_sep[episode][start:]
    class1 = clf1.predict(traj[:,:3])
    class2 = clf2.predict(traj)
    class3 = clf3.predict(traj)
    classes = list(zip(class2, class3))

    seq = []
    for t in range(len(class1)):
        if 1: seq.append(quad[classes[t]])
    seq = bcs.strip_rp(seq)
    new_trigrams = bcs.get_grams(seq, [1,2,3,4], n=3)
    trigrams = bcs.update_by_add(new_trigrams, trigrams)
trigrams = bcs.sort_by_val(trigrams)
print(list(trigrams.items())[:10])