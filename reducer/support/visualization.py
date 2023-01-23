import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from reducer.config import graphpath

def draw_unit_circle(ax):
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), "k--")

def color_time_2d(ax, x, y):
    T = len(x) - 2
    color = sns.color_palette("viridis", T)
    for t in range(T):
        ax.plot(x[t:t+2], y[t:t+2], color=color[t], alpha=0.5, marker="")

def color_time_3d(ax, x, y, z):
    T = len(x) - 2
    color = sns.color_palette("viridis", T)
    for t in range(T):
        ax.plot(x[t:t+2], y[t:t+2], z[t:t+2], color=color[t], alpha=0.5, marker="")

def common_label(fig, xlabel, ylabel):
    # Add a big axis, hide frame
    fig.add_subplot(111, frameon=False)

    # Hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def common_col_title(fig, titles, shape):
    N1, N2 = shape
    for n in range(N2):
        ax = fig.add_subplot(N1, N2, n+1, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax.set_title(titles[n])
    return fig

def savefig(figname="temp.png"):
    plt.savefig(os.path.join(graphpath, figname), dpi=200)

def plot_PCA_3d(data, figname="temp.png", save=True, plot_time=True):
    ax = plt.figure().add_subplot(projection="3d")
    pca = PCA(n_components=3)
    y_pca = pca.fit_transform(data)
    if plot_time: color_time_3d(ax, y_pca[:,0], y_pca[:,1], y_pca[:,2])
    else: ax.plot(y_pca[:,0], y_pca[:,1], y_pca[:,2], alpha=0.5)
    if save: plt.savefig(os.path.join(graphpath, figname))

def plot_trajectory(trajectory, figname="temp.png", save=True, plot_time=True, ax=None):
    if ax == None: ax = plt.figure().add_subplot(111)
    if plot_time: color_time_2d(ax, *trajectory)
    else: ax.plot(*trajectory, alpha=0.5)
    if save: plt.savefig(os.path.join(graphpath, figname))

def plot_multiple_trajectory(trajectories, figname="temp.png", save=True, plot_time=True):
    N = len(trajectories)
    fig = plt.figure(figsize=(N*3, 3))
    
    for c in range(N):
        ax = fig.add_subplot(1, N, c+1)
        plot_trajectory(trajectories[c], save=False, plot_time=plot_time, ax=ax)

    if save: plt.savefig(os.path.join(graphpath, figname))

def plot_multiple_trajectory2(trajectories, figname="temp.png", save=True, plot_time=True):
    N1 = len(trajectories); N2 = len(trajectories[0])
    fig = plt.figure(figsize=(N2*3, N1*3))
    
    for c in range(N1*N2):
        i, j = np.unravel_index(c, (N1, N2))
        ax = fig.add_subplot(N1, N2, c+1)
        plot_trajectory(trajectories[i][j], save=False, plot_time=plot_time, ax=ax)

    if save: plt.savefig(os.path.join(graphpath, figname))
    return plt.gcf()

