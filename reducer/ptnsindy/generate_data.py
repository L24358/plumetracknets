import os
import pickle
import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis
from reducer.config import modelpath

specify = 0

# Create "x"
if 0:
    # Load model
    rnn, inn, br, bi = bcs.model_loader(specify=specify)

    # Simulate from different observations
    all_data = np.zeros((1, 64))
    for tpe in ["constant", "switch", "noisy"]:
        for episode in range(240):
            dic = bcs.simulation_loader(specify, tpe, episode=episode)
            obs = dy.assigned_obs(dic["observations"])
            h_0 = dic["activities_rnn"][0]
            t, y = dy.sim(rnn, inn, br, bi, obs, h_0, T=len(dic["observations"]))
            all_data = np.vstack((all_data, y))
    all_data = all_data[1:]

    np.save(os.path.join(modelpath, "aesindy_results", f"data_from_real_agent={specify+1}.npy"), all_data)

# Create "dx"
all_data = np.load(os.path.join(modelpath, "aesindy_results", f"data_from_real_agent={specify+1}.npy"))
dic = {"x": all_data[1:], "dx": all_data[1:] - all_data[-1]}
f = open(os.path.join(modelpath, "aesindy_results", f"data_from_real_agent={specify+1}.pkl"), "wb")
pickle.dump(dic, f)

import pdb; pdb.set_trace()