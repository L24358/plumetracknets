import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.navigator as nav
from reducer.config import graphpath

def file_pointer(target, folderpath): # file_pointer in nav not working (also works differently)
    res = []
    for file in os.listdir(folderpath):
        if target in file: res.append(file)
    return res

obs_stack = np.zeros((1, 3))
modelnames = nav.file_finder()
for modelname in modelnames:
    print(modelname)
    foldername = os.path.splitext(modelname)[0]

    logfiles = file_pointer(".pkl", foldername) # get the first .pkl file
    print("Number of logfiles: ", len(logfiles))
    for logfile in logfiles:
        logpath = os.path.join(foldername, logfile)
        with open(logpath, 'rb') as f: log = pickle.load(f)

        for episode in range(len(log)):
            obs = np.asarray(log[episode]["observations"]).squeeze()
            obs_stack = np.vstack((obs_stack, obs))

fig = plt.figure(figsize=(9, 3))
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    bins = bcs.get_bins(obs_stack[:, i], 20)
    ax.hist(obs_stack[:, i], bins=bins, color="k")
    ax.set_title(f"Observation {i}")
    ax.set_ylabel("log(count)")
    ax.set_yscale("log")
plt.suptitle("Distribution of Observations")
plt.tight_layout()
plt.savefig(os.path.join(graphpath, "observation_distribution.png"))

print("Total number of observations: ", len(obs_stack))