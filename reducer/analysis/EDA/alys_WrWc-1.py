'''
Analysis of:
    1. Eigenvalues of the recurrent matrix W_r
    2. dot(eigenvectors, input matrix columns)
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.navigator as nav
import reducer.support.visualization as vis
from reducer.config import modelpath, graphpath

########## Part 1 ##########

# plot eigenvalues for recurrent matrix
modelnames = nav.file_finder(
        target="weight_hh",
        extension=".npy",
        parent_name=os.path.join(modelpath, "rnn_matrix"))

fig = plt.figure(figsize=(20, 4))

# collect info about eigenvalue, eigenvector and stability
df = pd.DataFrame() # columns: modelidx, evalue, if unstable, eigenvector
for i in range(5):
    rnn = np.load(modelnames[i])
    ev, ew = np.linalg.eig(rnn)

    ax = fig.add_subplot(1, 5, i+1)
    vis.draw_unit_circle(ax)

    idx_unstable = np.where(abs(ev) >= 1)[0]
    idx_stable = np.where(abs(ev) < 1)[0]
    for idx in range(64):
        dic = {"modelidx": i,
                "eigenvalue": ev[idx].item(),
                "abs(eigenvalue)": abs(ev[idx].item()),
                "unstable": abs(ev[idx]) >= 1,
                "eigenvector": ew[:, idx]}
        df = df.append(dic, ignore_index=True)

    ax.scatter(np.real(ev)[idx_stable], np.imag(ev)[idx_stable], color="b", label="stable")
    ax.scatter(np.real(ev)[idx_unstable], np.imag(ev)[idx_unstable], color="r", label="unstable")
    ax.legend()

plt.suptitle("Eigenvalues of Recurrent Connectivity Matrix")
plt.tight_layout()
plt.savefig(os.path.join(graphpath, "eigenvalue on unit circle.png"), dpi=200)
plt.clf()

########## Part 2 ##########

# load input matrix
modelnames = nav.file_finder(
        target="weight_ih",
        extension=".npy",
        parent_name=os.path.join(modelpath, "rnn_matrix"))
idic = {} # key: modelidx, value: input matrix
for i in range(5): idic[i] = np.load(modelnames[i])

# project input vectors onto eigenvectors
df_proj = pd.DataFrame()
for _, row in df.iterrows():
    proj = row["eigenvector"] @ idic[row["modelidx"]]
    proj_row = pd.Series(proj, index=["proj1", "proj2", "proj3"])
    new_row = row.append(proj_row)
    df_proj = df_proj.append(new_row, ignore_index=True)

# plot dot product
inp = "proj2" # which projection to plot
df_sub = df_proj[df["modelidx"] == 2]
df_unstable = df_sub[df_sub["unstable"]]
proj = df_sub[inp]
proj_unstable = df_unstable[inp]
location = np.where(df_sub["unstable"])[0]
fig = plt.figure(figsize=(4, 4))
idx_sort = list(np.argsort(abs(proj)))

plt.plot(np.array(abs(proj))[idx_sort], "b.")
plt.plot([idx_sort.index(loc) for loc in location],
        abs(proj_unstable), "r.")
plt.savefig(os.path.join(graphpath, "temp.png"))

print(df_unstable)
print(abs(proj_unstable))
