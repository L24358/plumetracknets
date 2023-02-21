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

specify = 0
episode = 16
plot_Cfp = False
plot_vfp = True
plot_theta = False

# load
plane_dic = bcs.pklload("pcadist", f"planes_agent={specify+1}.pkl")
pca_dic64 = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
all_traj_sep = pca_dic64["all_traj_sep"]
X1, Y1, Z1 = plane_dic["Cfp"]
X2, Y2, Z2 = plane_dic["vfp"]
X3, Y3, Z3 = plane_dic["theta"]

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
if plot_Cfp: ax.plot_surface(X1, Y1, Z1, alpha=0.5, shade="r")
if plot_vfp: ax.plot_surface(X2, Y2, Z2, alpha=0.5, shade="g")
if plot_theta: ax.plot_surface(X3, Y3, Z3, alpha=0.5, shade="b")
traj = all_traj_sep[episode][:, :3]
ax = vis.plot_trajectory(traj.T, save=False, ax=ax)
vis.gen_gif(True, f"svc_traj_agent={specify+1}_episode={episode}", ax, stall=5, angle1=15, angles=None)