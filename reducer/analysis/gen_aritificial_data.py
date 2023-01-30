import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.visualization as vis
from reducer.config import modelpath
from reducer.support.basics import constant, single_sine, combine_dict

specify = 0
episodes = [3, 4, 5, 10, 13, 15, 18, 30, 35, 40, 42, 43, 45, 47, 48, 50, 53, 55, 58, 60, 63, 65]

data = {"f": [], "phi": [], "b_C": [], "b_y": [], "b_x": []}

keys = ["C", "y", "x"]
for episode in episodes:
    with open(os.path.join(modelpath, "fit", f"agent={specify+1}_episode={episode}_manual.pkl"), "rb") as f:
        temp = pickle.load(f)
        
        f_C, phi_C, b_C = temp["C"][0][1:4]
        f_y, phi_y, b_y = temp["y"][0][1:4]
        data["f"] += [f_C, f_y]
        data["phi"].append(phi_C - phi_y) # -4, 2, 0.5
        data["b_C"].append(b_C)
        data["b_y"].append(b_y)

        if temp["x"][1] == constant: data["b_x"].append(temp["x"][0][0]) # -0.2
         
f_mean, f_std = np.mean(data["f"]), np.std(data["f"])
bC_mean, bC_std = np.mean(data["b_C"]), np.std(data["b_C"])
by_mean, by_std = np.mean(data["b_y"]), np.std(data["b_y"])

N = sys.argv[1]
