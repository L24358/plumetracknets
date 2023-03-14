import os
import numpy as np

for _ in range(100):
    seed = np.random.randint(0, high=99999)
    noise_perc = 0.0
    save = 0
    os.system(f"python3 fit_pca-sindyc_d=3.py {str(seed)} {str(noise_perc)} {str(save)}")