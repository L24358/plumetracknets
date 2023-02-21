"""
Nonlinear SVM on odor concentration fixed points (Cfp) and wind concentration fixed points (vfp).

@ notes:
    - LinearSVC on theta (in -4.py) score, tol=0.0: 0.587
    - LinearSVC on Cfp used on theta score: 0.588
    - rbf on theta (in -4.py) score, tol=0.0: 0.642
    - rbf on theta (in -4.py) score, tol=0.2 (~ mean value): 0.715
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
plot_Cfp = True
plot_vfp = True
plot_theta = True
plot_theta_dots = False
plot_vfp_dots = False

# load results
pca_dic64 = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca64 = pca_dic64["pca"]
pCfp = bcs.pklload("pcadist", f"Cfp_pos_agent={specify+1}.pkl") # positive (odor) concentration fixed points
nCfp = bcs.pklload("pcadist", f"Cfp_neg_agent={specify+1}.pkl") # negative
pvfp = bcs.pklload("pcadist", f"vfp_pos_agent={specify+1}_save=all.pkl") # positive (wind) velocity fixed points
nvfp = bcs.pklload("pcadist", f"vfp_neg_agent={specify+1}_save=all.pkl") # negative
ptheta= bcs.pklload("pcadist", f"theta_pos_agent={specify+1}_save=all.pkl")
ntheta = bcs.pklload("pcadist", f"theta_neg_agent={specify+1}_save=all.pkl")
ptval = bcs.npload("pcadist", f"thetaval_pos_agent={specify+1}.npy")
ntval = bcs.npload("pcadist", f"thetaval_neg_agent={specify+1}.npy")

# concat value and assign values
pCfp = bcs.concat_dic_values(pCfp) # extract and concat all values from dict
nCfp = bcs.concat_dic_values(nCfp)
pvfp = bcs.concat_dic_values(pvfp) # extract and concat all values from dict
nvfp = bcs.concat_dic_values(nvfp)
ptheta = bcs.concat_dic_values(ptheta) # extract and concat all values from dict
ntheta = bcs.concat_dic_values(ntheta)
lpCfp = [1]* len(pCfp) # labels for pCfp
lnCfp = [0]* len(nCfp)
lpvfp = [1]* len(pvfp) # labels for pCfp
lnvfp = [0]* len(nvfp)
lptheta = [1]* len(ptheta)
lntheta = [0]* len(ntheta)
data1 = np.array(pCfp + nCfp)
label1 = np.array(lpCfp + lnCfp)
data2 = np.array(pvfp + nvfp)
label2 = np.array(lpvfp + lnvfp)
data3 = np.array(ptheta + ntheta)
label3 = np.array(lptheta + lntheta)

# perform SVM
clf1 = LinearSVC()
clf1.fit(data1, label1)
print("Score for concentration: ", clf1.score(data1, label1))
clf2 = LinearSVC()
data2 = pca64.transform(data2) # Needs pca.transform because some idiot did not save the transformed version
clf2.fit(data2, label2)
print("Score for wind velocity: ", clf2.score(data2, label2))
clf3 = LinearSVC()
clf3.fit(data3, label3) # Doesn't need pca.transform because the saved data is already transformed
print("Score for agent angle: ", clf3.score(data3, label3))

# save
svc_dic = {"Cfp": clf1, "vfp": clf2, "theta": clf3}
bcs.pklsave(svc_dic, "pcadist", f"svc_agent={specify+1}.pkl")

# pca transform hyperplane
def Z0(clf, X, Y):
    return (-clf.intercept_[0]-clf.coef_[0][0]*X-clf.coef_[0][1]*Y) / clf.coef_[0][2]

def Zn(clf, X, Y, means):
    """Get hyperplane, using mean values for the rest of the coordinates."""
    coef, intercept = clf.coef_[0], clf.intercept_[0]
    assert len(means) == len(coef) - 3
    res = -intercept - coef[0]*X - coef[1]*Y - np.dot(coef[3:], means)
    return res / coef[2]

# extract plane
minx, maxx = min(data1.T[0])-1, max(data1.T[0])+1
miny, maxy = min(data1.T[1])-1, max(data1.T[1])+1
X1, Y1 = np.meshgrid(np.linspace(minx, maxx, 20), np.linspace(miny, maxy, 20))
Z1 = Z0(clf1, X1, Y1)

# extract hyperplane for wind velocity
minx, maxx = min(data2.T[0])-1, max(data2.T[0])+1
miny, maxy = min(data2.T[1])-1, max(data2.T[1])+1
X2, Y2 = np.meshgrid(np.linspace(minx, maxx, 20), np.linspace(miny, maxy, 20))
Z2 = Zn(clf2, X2, Y2, np.mean(data2, axis=0)[3:])

# extract hyperplane for theta
minx, maxx = min(data3.T[0])-1, max(data3.T[0])+1
miny, maxy = min(data3.T[1])-1, max(data3.T[1])+1
X3, Y3 = np.meshgrid(np.linspace(minx, maxx, 20), np.linspace(miny, maxy, 20))
Z3 = Zn(clf3, X3, Y3, np.mean(data3, axis=0)[3:])

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
if plot_Cfp: ax.plot_surface(X1, Y1, Z1, alpha=0.5, shade="r")
if plot_vfp: ax.plot_surface(X2, Y2, Z2, alpha=0.5, shade="g")
if plot_theta: ax.plot_surface(X3, Y3, Z3, alpha=0.5, shade="b")
if plot_theta_dots:
    ax.scatter(*np.array(ptheta).T, c=ptval, cmap="Reds", s=3)
    ax.scatter(*np.array(ntheta).T, c=ntval, cmap="Greens", s=3)
if plot_vfp_dots: # for sanity check on hyperplane extraction
    ax.scatter(*np.array(pca64.transform(pvfp)).T[:3], color="c", s=1, alpha=0.2)
    ax.scatter(*np.array(pca64.transform(nvfp)).T[:3], color="m", s=1, alpha=0.2)
vis.gen_gif(True, f"pcadist_all_linearsvc", ax, stall=5, angle1=30, angles=None)

# save
plane_dic = {"Cfp": [X1, Y1, Z1], "vfp": [X2, Y2, Z2], "theta": [X3, Y3, Z3]}
bcs.pklsave(plane_dic, "pcadist", f"planes_agent={specify+1}.pkl")
