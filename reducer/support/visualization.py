import os
import tqdm
import imageio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.dynamics as dy
from sklearn.decomposition import PCA
from reducer.config import graphpath

########################################################
#                     Plot Singles                     #
########################################################

def plot_PCA_3d(data, figname="temp.png", save=True, plot_time=True, ax=None, **kwargs):
    """
    Plot high-dimensional `data` in 3 dims using PCA.
    @ Args:
        - data: np.ndarray, shape=(observations, features)
        - figname: str, default="temp.png"
        - save: bool, save figure, default=True
        - plot_time: bool, plot time in color, default=True
    """
    kw = {"color": "k", "xlabel": "", "ylabel": "", "subtitle": ""}
    kw.update(kwargs)

    if ax == None: ax = plt.figure().add_subplot(projection="3d")
    pca = PCA(n_components=3)
    y_pca = pca.fit_transform(data)
    if plot_time: color_time_3d(ax, y_pca[:,0], y_pca[:,1], y_pca[:,2])
    else: ax.plot(y_pca[:,0], y_pca[:,1], y_pca[:,2], alpha=0.5, color=kw["color"])
    
    ax.set_xlabel(kw["xlabel"]); ax.set_ylabel(kw["ylabel"]); ax.set_title(kw["subtitle"])
    if save: savefig(figname)

def plot_trajectory(trajectory, figname="temp.png", save=True, plot_time=True, ax=None, **kwargs):
    """
    Plot `trajectory` in `ax`.
    @ Args:
        - trajectory: array-like, shape=(2, N), N=#data
        - ax: axis to plot in, default=None
    @ Kwargs:
        - color: str, default="k"
    """
    kw = {"color": "k", "xlabel": "x", "ylabel": "y", "subtitle": "", "projection": "2d"}
    kw.update(kwargs)
    projection = kw["projection"]

    if ax == None: ax = plt.figure().add_subplot(111, projection=projection)
    if plot_time and projection == "2d": color_time_2d(ax, *trajectory)
    elif plot_time and projection == "3d": color_time_3d(ax, *trajectory)
    else: ax.plot(*trajectory, alpha=0.5, color=kw["color"])
    ax.set_xlabel(kw["xlabel"]); ax.set_ylabel(kw["ylabel"]); ax.set_title(kw["subtitle"])
    if save: savefig(figname)
    return ax

def plot_quantities(quantities, figname="temp.png", save=True, ax=None, **kwargs):
    """
    Plot `quantities` in `ax`.
    """
    Q, N = len(quantities), len(quantities[0])
    kw = {"color": ["k"]*Q, "time": range(N), "label": [""]*Q, "xlabel":"time", "ylabel":"", "subtitle":"", "linestyle":["-"]*Q}
    kw.update(kwargs)

    if ax == None: ax = plt.figure().add_subplot(111)
    for q in range(len(quantities)):
        ax.plot(kw["time"], quantities[q], color=kw["color"][q], label=kw["label"][q], linestyle=kw["linestyle"][q])
        ax.set_xlabel(kw["xlabel"]); ax.set_ylabel(kw["ylabel"]); ax.set_title(kw["subtitle"])
        if not (kw["label"] == [""]*Q): ax.legend()
    if save: savefig(figname)

########################################################
#                    Plot Multiples                    #
########################################################

def plot_multiple_trajectory(trajectories, figname="temp.png", save=True, plot_time=True, **kwargs):
    """
    Plot `trajectories` in multiple subplots of shape (1, s), s=#subplots.
    @ Args:
        - trajectories: array-like, shape=(s, 2, N)
    @ Kwargs:
        - xlabel, ylabel, subtitle: list, shape=(s,), default=[""]*s
        - suptitle: str, default=""
        - color, str, default="k"
    """
    s = len(trajectories)
    kw = {"xlabel": [""]*s, "ylabel": [""]*s, "subtitle": [""]*s, "suptitle": "", "color": "k"}
    kw.update(kwargs)

    fig = plt.figure(figsize=(s*3, 3))
    
    for c in range(s):
        ax = fig.add_subplot(1, s, c+1)
        plot_trajectory(trajectories[c], save=False, plot_time=plot_time, ax=ax, **kw)
        ax.set_xlabel(kw["xlabel"][c]); ax.set_ylabel(kw["ylabel"][c]); ax.set_title(kw["subtitle"][c])
    plt.suptitle(kw["suptitle"])

    if save: savefig(figname)

def plot_multiple_trajectory2(trajectories, figname="temp.png", save=True, plot_time=True):
    """
    Plot `trajectories` in multiple subplots of shape (N1, N2) = trajectories.shape.
    @ Args:
        - trajectories: array-like, shape=(N1, N2)
    """
    N1 = len(trajectories); N2 = len(trajectories[0])
    fig = plt.figure(figsize=(N2*3, N1*3))
    
    for c in range(N1*N2):
        i, j = np.unravel_index(c, (N1, N2))
        ax = fig.add_subplot(N1, N2, c+1)
        plot_trajectory(trajectories[i][j], save=False, plot_time=plot_time, ax=ax)

    if save: savefig(figname, clear=False)
    return plt.gcf()

def plot_multiple_quantities2(trajectories, figname="temp.png", save=True, override_N=None, **kwargs):
    """
    Plot `quantities` in multiple subplots of shape (N1, N2) = trajectories.shape.
    @ Args:
        - quantities: array-like, shape=(N1, N2, q), q=#quantities
    """
    N1 = len(trajectories); N2 = len(trajectories[0])
    N = N1*N2 if override_N == None else override_N
    kw = {"subtitle": [""]*N, "xlabel": ["time"]*N, "title": "", "color":["k"]*N}
    kw.update(kwargs)

    double = lambda x: [x, x]
    fig = plt.figure(figsize=(N2*3, N1*3))
    for c in range(N):
        i, j = np.unravel_index(c, (N1, N2))
        ax = fig.add_subplot(N1, N2, c+1)
        plot_quantities(trajectories[i][j], save=False, ax=ax, xlabel=kw["xlabel"][c], subtitle=kw["subtitle"][c],\
        color=double(kw["color"][c]), linestyle=["-", "--"]) # TODO: VERY temporary fix for color, linestyle

    plt.suptitle(kw["title"])
    if save: savefig(figname, clear=False)
    return plt.gcf()

def plot_multiple_hist2(trajectories, figname="temp.png", save=True, override_N=None, **kwargs):
    """
    Plot `quantities` in multiple subplots of shape (N1, N2) = trajectories.shape.
    @ Args:
        - quantities: array-like, shape=(N1, N2, q), q=#quantities
    """
    N1 = len(trajectories); N2 = len(trajectories[0])
    N = N1*N2 if override_N == None else override_N
    kw = {"subtitle": [""]*N, "xlabel": [""]*N, "ylabel": ["time"]*N, "title": "", "color":["k"]*N}
    kw.update(kwargs)

    fig = plt.figure(figsize=(N2*3, N1*3))
    for c in range(N):
        i, j = np.unravel_index(c, (N1, N2))
        ax = fig.add_subplot(N1, N2, c+1)
        ax.hist(trajectories[i][j])
        ax.set_title(kw["subtitle"][c])
        ax.set_xlabel(kw["xlabel"][c])
        ax.set_ylabel(kw["ylabel"][c])

    plt.suptitle(kw["title"])
    if save: savefig(figname, clear=False)
    return plt.gcf()

########################################################
#                   Plot Specifics                     #
########################################################

def plot_obs_act_traj(actions, observations, figname="temp.png"):
    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(131)
    plot_quantities(observations.T, save=False, ax=ax1, ylabel="value", subtitle="Observations", color=["k", "r", "b"], label=["C", "y", "x"])

    ax2 = fig.add_subplot(132)
    plot_quantities(actions.T, save=False, ax=ax2, ylabel="value", subtitle="Actions", color=["g", "m"], label=["r", "\u03B8"])

    ax3 = fig.add_subplot(133)
    traj = dy.get_trajectory(actions)
    plot_trajectory(traj.T, save=False, ax=ax3)

    savefig(figname=figname)

def gen_gif(gen_fig, foldername, ax, stall=5, angle1=30):
    if not os.path.exists(os.path.join(graphpath, foldername)): os.mkdir(os.path.join(graphpath, foldername))
    if gen_fig:
        for angle in tqdm.tqdm(np.linspace(-180, 180, 20)):
            ax.view_init(angle1, angle)
            savefig(figname=f"{foldername}/{round(angle,0)}.png", clear=False)

    images = []
    for angle in tqdm.tqdm(np.linspace(-180, 180, 20)):
        for _ in range(stall):
            images.append(imageio.imread(os.path.join(graphpath, foldername, f"{round(angle,0)}.png")))
    imageio.mimsave(os.path.join(graphpath, f"{foldername}.gif"), images)

########################################################
#                  Helper Functions                    #
########################################################

def draw_unit_circle(ax):
    """Plots a dashed unit circle on the axis `ax`."""
    t = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), "k--")

def color_time_2d(ax, x, y):
    """Colors the (`x`, `y`) trajectory by time on the axis `ax` (2D graph)."""
    T = len(x) - 2
    color = sns.color_palette("viridis", T)
    for t in range(T):
        ax.plot(x[t:t+2], y[t:t+2], color=color[t], alpha=0.5, marker="")

def color_time_3d(ax, x, y, z):
    """Colors the (`x`, `y`) trajectory by time on the axis `ax` (3D graph)."""
    T = len(x) - 2
    color = sns.color_palette("viridis", T)
    for t in range(T):
        ax.plot(x[t:t+2], y[t:t+2], z[t:t+2], color=color[t], alpha=0.5, marker="")

def common_label(fig, xlabel, ylabel):
    """Put a common `xlabel`, `ylabel` on the figure `fig`."""
    # Add a big axis, hide frame
    fig.add_subplot(111, frameon=False)

    # Hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def common_col_title(fig, titles, shape):
    """Put a common `title` on the figure `fig`."""
    N1, N2 = shape
    for n in range(N2):
        ax = fig.add_subplot(N1, N2, n+1, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        ax.set_title(titles[n])
    return fig

def savefig(figname="temp.png", clear=True, dpi=200):
    """Saves figure in `graphpath` as specified in config.py."""
    plt.tight_layout()
    plt.savefig(os.path.join(graphpath, figname), dpi=dpi)
    if clear: plt.clf()

