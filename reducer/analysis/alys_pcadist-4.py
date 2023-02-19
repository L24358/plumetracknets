"""
SVM on agent angle (theta) and odor concentration fixed points (Cfp).
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
plane = "Cfp" # Cfp or theta
dots = "theta" # Cfp or theta or vfp
tol = 0.3

# load results
pCfp = bcs.pklload("pcadist", f"{plane}_pos_agent={specify+1}.pkl") # positive (odor) concentration fixed points
nCfp = bcs.pklload("pcadist", f"{plane}_neg_agent={specify+1}.pkl") # negative
pvfp = bcs.pklload("pcadist", f"{dots}_pos_agent={specify+1}.pkl") # positive (wind) velocity fixed points
nvfp = bcs.pklload("pcadist", f"{dots}_neg_agent={specify+1}.pkl") # negative

# concat value
pCfp = bcs.concat_dic_values(pCfp) # extract and concat all values from dict
nCfp = bcs.concat_dic_values(nCfp)
pvfp = bcs.concat_dic_values(pvfp) # extract and concat all values from dict
nvfp = bcs.concat_dic_values(nvfp)

# for theta 
if plane == "theta":
    pval = bcs.npload("pcadist", f"thetaval_pos_agent={specify+1}.npy")
    nval = bcs.npload("pcadist", f"thetaval_neg_agent={specify+1}.npy")
    new_pCfp, new_nCfp = [], []
    for i in range(len(pCfp)):
        if pval[i] > tol: new_pCfp.append(pCfp[i])
    for i in range(len(nCfp)):
        if abs(pval[i]) > tol: new_nCfp.append(nCfp[i])
    pCfp, nCfp = new_pCfp, new_nCfp

# assign labels
lpCfp = [1]* len(pCfp) # labels for pCfp
lnCfp = [2]* len(nCfp)
data = np.array(pCfp + nCfp)
label = np.array(lpCfp + lnCfp)

# perform SVM
clf = SVC(gamma="auto")
clf.fit(data, label)
print("Score: ", clf.score(data, label))

# extract "linear" version of nonlinear plane
Z = lambda X, Y: (-clf.intercept_[0]-clf.coef_[0][0]*X-clf.coef_[0][1]*Y) / clf.coef_[0][2]
minx, maxx = min(data.T[0])-1, max(data.T[1])+1
miny, maxy = min(data.T[1])-1, max(data.T[1])+1
X, Y = np.meshgrid(np.linspace(minx, maxx, 20), np.linspace(miny, maxy, 20))

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z(X, Y), alpha=0.5)
ax.scatter(*np.array(pvfp).T, color="c", s=1, alpha=0.2)
ax.scatter(*np.array(nvfp).T, color="m", s=1, alpha=0.2)
vis.gen_gif(True, f"pcadist_{plane}_{dots}", ax, stall=5, angle1=30, angles=None)