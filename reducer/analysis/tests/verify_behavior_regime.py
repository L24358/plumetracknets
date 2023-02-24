"""
@ reference:
    - multicolored line: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
"""

import os 
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

specify = 0
tpe = "constant"
c1, c2 = 12, 25
regime = bcs.pklload("regimes", f"regime_agent={specify+1}_criterion=({c1},{c2}).pkl")

for episode in range(240):
    dic = bcs.simulation_loader(specify, tpe, episode=episode)
    actions = dic["actions"]
    traj, _, _ = dy.get_trajectory2(actions) # already transformed actions

    # get color
    color = []
    for t in range(len(traj)-1):
        if regime[episode]["tracking"][t]: color.append(2)
        elif regime[episode]["recovery"][t]: color.append(0)
        elif regime[episode]["lost"][t]: color.append(-2)
        else: raise bcs.AlgorithmError(f"Time step {t} of episode {episode} does not belong to any regime.")

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6,4))
    vis.plot_trajectory_2d(traj.T, save=False, ax=axs[0])

    plt.plot(*traj.T, color="white")
    x, y = traj.T
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = ListedColormap(['r', 'b', 'g'])
    norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(color)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    fig.colorbar(line, ax=axs[1])
    vis.savefig(figname=bcs.fjoin(f"behavior_regime_criterion=({c1},{c2})", f"agent={specify+1}_episode={episode}.png", tpe=2))