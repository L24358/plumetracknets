import numpy as np
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# Load model
specify = 0
data = bcs.pklload("pca_frame", f"stacked_agent={specify+1}.pkl")
observations = data["observations"]

# unique values
if 0:
    obs_round = np.around(observations, 1)
    obs_tuple = [tuple(l) for l in obs_round]
    obs_unique = list(set(obs_tuple))
    bcs.npsave(np.array(obs_unique), "observations", "unique.npy")

# random selection
idxs = np.random.choice(range(len(observations)), size=5000, replace=False)
bcs.npsave(observations[idxs], "observations", "random_select.npy")
