import numpy as np
import reducer.support.basics as bcs
import reducer.support.dynamics as dy
import reducer.support.visualization as vis

# Load model
rnn, inn, br, bi = bcs.model_loader(specify="random") # Take the first model

# Run simulation
x_0 = [5, 0, 0]
h_0 = np.random.uniform(low=-1, high=1, size=64) # random initial hidden states
t, y = dy.sim(rnn, inn, br, bi, dy.constant_obs(x_0), h_0, T=1000)
vis.plot_PCA_3d(y, plot_time=False) # Plot first 3 PCA dimensions
