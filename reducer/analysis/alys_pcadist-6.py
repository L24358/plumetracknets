"""
Nonlinear SVM on odor concentration fixed points (Cfp) and wind concentration fixed points (vfp).
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

# hypoerparameters
specify = 0

# load results
pca_dic = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca = pca_dic["pca"]
pCfp = bcs.pklload("pcadist", f"Cfp_pos_agent={specify+1}.pkl") # positive (odor) concentration fixed points
nCfp = bcs.pklload("pcadist", f"Cfp_neg_agent={specify+1}.pkl") # negative
pvfp = bcs.pklload("pcadist", f"vfp_pos_agent={specify+1}_save=all.pkl") # positive (wind) velocity fixed points
nvfp = bcs.pklload("pcadist", f"vfp_neg_agent={specify+1}_save=all.pkl") # negative

# concat value and assign values
pCfp = bcs.concat_dic_values(pCfp) # extract and concat all values from dict
nCfp = bcs.concat_dic_values(nCfp)
pvfp = bcs.concat_dic_values(pvfp) # extract and concat all values from dict
nvfp = bcs.concat_dic_values(nvfp)
lpCfp = [1]* len(pCfp) # labels for pCfp
lnCfp = [2]* len(nCfp)
lpvfp = [1]* len(pvfp) # labels for pCfp
lnvfp = [2]* len(nvfp)
data1 = np.array(pCfp + nCfp)
label1 = np.array(lpCfp + lnCfp)
data2 = np.array(pvfp + nvfp)
label2 = np.array(lpvfp + lnvfp)

# perform SVM
clf1 = LinearSVC()
clf1.fit(data1, label1)
print("Score for concentration: ", clf1.score(data1, label1))
clf2 = LinearSVC()
data2_pca = pca.transform(data2)[:,:7]
clf2.fit(data2_pca, label2)
print("Score for wind velocity: ", clf2.score(data2_pca, label2))

# pca transform hyperplane
def Z(clf, X, Y):
    return (-clf.intercept_[0]-clf.coef_[0][0]*X-clf.coef_[0][1]*Y) / clf.coef_[0][2]

def Zn(clf, Xs):
    """Get hyperplane."""
    res = np.ones(Xs[0].shape)*(-clf.intercept_[0])
    for i in range(len(Xs)-1):
        res -= clf.coef_[0][i]*Xs[i]
    return res / clf.coef_[0][-1]

xx = data2.T # use data2 as grid points
zz = Zn(clf2, xx)
xz = np.hstack([xx[:-1].T, zz.reshape(-1, 1)])
xz_pca = pca.transform(xz)
