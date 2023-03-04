"""
Plots a histogram for the amount of fixed points.

TODO:
    Should include cases where there are no fixed points.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# hyperparameters
specify = 0
tpe = "constant"
to_save = "pca64"

# load model
info = bcs.npload("pcadist", f"fppcainfo_agent={specify+1}_save={to_save}.npy")

# main
ts = info.T[-1]
prev_t = None
counts, agg = [], 1 # number of fixed point counts, aggregation count
for i in range(len(ts)):
    if ts[i] != prev_t:
        counts.append(agg)
        agg = 1 # reset
    else: agg += 1
    prev_t = ts[i]

# plot
ax = plt.figure().add_subplot(111)
plt.hist(counts, color="b")
plt.xlabel("fixed point counts"); plt.ylabel("counts")
vis.simpleaxis(ax)
vis.savefig(figname=f"fp_count_agent={specify+1}.png")