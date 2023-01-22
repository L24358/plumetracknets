import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.navigator as nav
from reducer.config import modelpath, graphpath

# Extract (save as npy) RNN matrices / actor / dist
modelnames = nav.file_finder()
for modelname in modelnames:
    print(modelname)
    os.system(f"python3 /src/tracer/ppo/evalCli.py --model_fname {modelname}")

# Move them into /data/actor/
files = nav.file_finder(target="_type=", extension=".npy", parent_name=modelpath)
for file in files: os.rename(file, os.path.join(modelpath, "actors", os.path.basename(file)))