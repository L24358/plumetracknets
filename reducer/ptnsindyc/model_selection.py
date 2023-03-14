import os
import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.visualization as vis

# hyperparameters
specify = 0
episode = 5

# read file
action = lambda row: bcs.to_(bcs.split_sort(row), [int, float, float, float])
res = bcs.read_row(f"agent={specify+1}_episode={episode}_d=3.txt", "ptnsindyc_model_selection", action)

for row in res:
    seed, noise_perc, err_train, err_test = row
    if (err_train < 0.1) and (err_test < 0.8):
        os.system(f"python3 fit_pca-sindyc_d=3.py {str(int(seed))} {str(noise_perc)} {str(1)}")