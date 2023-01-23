import os
import pickle
import numpy as np
import reducer.support.basics as bcs
import reducer.support.navigator as nav
from reducer.config import modelpath

def file_pointer(target, folderpath): # file_pointer in nav not working (also works differently)
        res = []
        for file in os.listdir(folderpath):
            if target in file: res.append(file)
        return res

for item in []: # ["observations", "actions"]
    modelnames = nav.file_finder()
    for modelname in modelnames:
        print(modelname)
        foldername = os.path.splitext(modelname)[0]

        logfiles = file_pointer(".pkl", foldername)
        print("Number of logfiles: ", len(logfiles))
        for logfile in logfiles:
            logpath = os.path.join(foldername, logfile)
            with open(logpath, 'rb') as f: log = pickle.load(f)

            seed = bcs.param_finder(os.path.splitext(modelname)[0], "seed", sep2=None)[4:]
            tpe = os.path.splitext(logfile)[0]
            for episode in range(len(log)):
                obs = np.asarray(log[episode][item]).squeeze()
                np.save(os.path.join(modelpath, item, f"seed={seed}_tpe={tpe}_episode={episode}.npy"), obs)

dic = {"rnn_hxs": "activities_rnn", "hx1_actor": "activities_MLP1"}
for item in ["rnn_hxs", "hx1_actor"]: 
    modelnames = nav.file_finder()
    for modelname in modelnames:
        print(modelname)
        foldername = os.path.splitext(modelname)[0]

        logfiles = file_pointer(".pkl", foldername)
        print("Number of logfiles: ", len(logfiles))
        for logfile in logfiles:
            logpath = os.path.join(foldername, logfile)
            with open(logpath, 'rb') as f: log = pickle.load(f)

            seed = bcs.param_finder(os.path.splitext(modelname)[0], "seed", sep2=None)[4:]
            tpe = os.path.splitext(logfile)[0]
            for episode in range(len(log)):
                activities = log[episode]["activity"]
                obs_stack = [np.asarray(activities[t][item]).squeeze() for t in range(len(activities))]
                obs_stack = np.vstack(obs_stack)
                np.save(os.path.join(modelpath, dic[item], f"seed={seed}_tpe={tpe}_episode={episode}.npy"), obs_stack)