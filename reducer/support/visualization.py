import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from reducer.config import graphpath

def draw_unit_circle(ax):
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), "k--")

def color_time_3d(ax, x, y, z):
    T = len(x) - 2
    color = sns.color_palette("viridis", T)
    for t in range(T):
        ax.plot(x[t:t+2], y[t:t+2], z[t:t+2], color=color[t], alpha=0.5, marker="")

def plot_PCA_3d(data, figname="temp.png", save=True, plot_time=True):
    ax = plt.figure().add_subplot(projection="3d")
    pca = PCA(n_components=3)
    y_pca = pca.fit_transform(data)
    if plot_time: color_time_3d(ax, y_pca[:,0], y_pca[:,1], y_pca[:,2])
    else: ax.plot(y_pca[:,0], y_pca[:,1], y_pca[:,2], alpha=0.5)
    if save: plt.savefig(os.path.join(graphpath, figname))