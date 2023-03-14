import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from sklearn.decomposition import PCA

# hyperparameters
specify = 0
tpe = "constant"
episode = 16

# load
rnn, inn, br, bi = bcs.model_loader(specify=specify)
dic = bcs.simulation_loader(specify, tpe, episode=episode)
obs = dic["observations"]
acts = dic["actions"]
hs = dic["activities_rnn"]
pca_dic64 = bcs.pklload("pca_frame", f"pcaskl_agent={specify+1}_n=64.pkl")
pca64 = pca_dic64["pca"]

# main
obs_sffl = obs.copy()
np.random.shuffle(obs_sffl)
_, hs_sffl, acts_sffl = dy.sim_actor(rnn, inn, br, bi, dy.assigned_obs(obs_sffl), np.zeros(64,), specify, T=len(obs))

# pca
pca = PCA(n_components=64)
pca.fit(hs_sffl)
print(np.cumsum(pca.explained_variance_ratio_)[:10])
print(np.cumsum(pca64.explained_variance_ratio_)[:10])